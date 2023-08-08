import wfdb
import numpy as np
import glob, os
import pandas as pd
import ast


path_to_dir = "arrythmia_db"

def generate_all_symbol_set():

     symbol_set = set()
     for direction in glob.glob(os.path.join(path_to_dir, '*.dat')):  
          path, full_name = os.path.split(direction)
          # split filename and extension
          filename, _ = full_name.split('.')

          # read in annotation
          annotation = wfdb.rdann(os.path.join(path, filename), 'atr')
          for ele in annotation.symbol:

               symbol_set.add(ele)

     with open(os.path.join('arrythmia_csv', 'unique_symbol_in_arrythmia_DB.txt'), 'w') as f:
          f.write(str([ele for ele in symbol_set]))
          f.close()          

def data_segmentation():
     N = ['N', 'L', 'R', 'j', 'e', '.']
     S = ['A', 'a', 'S', 'J']
     V = ['!', 'V', 'E', '[', ']']
     F = ['F']
     Q = ['f', '/', 'Q']
     non_noisy_symbols = ['N', '[', 'a', 'S', 'E', 'j', 'R', '+', 'V', 'J', 'x', ']', '"', 'A', 'e', '!', '~', 'F', 'L', '.']
     non_heart_beat_but_ok_symbols = ['+', '~']

     for direction in glob.glob(os.path.join(path_to_dir, '*.dat')):
          path, full_name = os.path.split(direction)
          # split filename and extension
          filename, _ = full_name.split('.')

          # read in record 
          record = wfdb.rdrecord(os.path.join(path, filename))

          # read in annotation
          annotation = wfdb.rdann(os.path.join(path, filename), 'atr')

          # Number of samples in each 10-second segment
          segment_length = 10 * record.fs

          num_chunk = len(record.p_signal) // segment_length

          # Segment the signal and create labels
          segments = []
          labels = []




          for i in range(num_chunk):
                   # Get the annotation symbols for the beats in this segment
               start = i * segment_length
               end = (i + 1) * segment_length
               beats_in_segment = (annotation.sample >= start) & (annotation.sample < end)
               beat_symbols = np.array(annotation.symbol)[beats_in_segment]

               temp_label = ''
               if all(element in non_noisy_symbols for element in beat_symbols):
                    if all(element in N+non_heart_beat_but_ok_symbols for element in beat_symbols):
                         temp_label = 'Normal'
                    else:
                         temp_label = "arrythmia"

               else:
                    temp_label = 'junk'

               segments.append([start, end])
               labels.append(temp_label)

          # Create a DataFrame and export to CSV
          df = pd.DataFrame({
          'beat_idx': np.arange(len(segments)),
          'beat_range': segments,
          'classification': labels
          })

          df.to_csv(f'arrythmia_csv/{filename}_segmentation.csv', index=False)


def data_statistics(path_to_dir):
     N = ['N', 'L', 'R', 'j', 'e', '.']
     S = ['A', 'a', 'S', 'J']
     V = ['!', 'V', 'E', '[', ']']
     F = ['F']
     Q = ['f', '/', 'Q']

     # set up counter
     total_length = 0
     N_counter = 0
     arrythmia_counter = 0
     Q_counter = 0

     unique_key = set()
     for direction in glob.glob(os.path.join(path_to_dir, '*.csv')):
          df = pd.read_csv(direction)
          data_mat = df.values
          symbol = data_mat[:, -1]
          total_length = total_length + len(data_mat)
          N_counter = N_counter + len([ele for ele in symbol if ele == 'Normal'])
          arrythmia_counter = arrythmia_counter + len([ele for ele in symbol if ele == 'arrythmia'])          
          Q_counter = Q_counter + len([ele for ele in symbol if ele == 'junk'])
          for ele in symbol:
               unique_key.add(ele)

     with open(os.path.join(path_to_dir, f'{os.path.split(path_to_dir)[-1]}_data_stats.txt'), 'w') as f:

          f.write(f"total number of total beats is: {total_length} \n")
          f.write(f"total number of Normal pieces (N) is : {N_counter} \n")
          f.write(f"total number of junk pieces (junk) is : {Q_counter} \n")
          f.write(f"total number of arrythmiam pieces (S, V, F) is : {arrythmia_counter} \n")
          f.write(str([ele for ele in unique_key]))
          f.close()


def file_check(path_to_csv):
  for direction in glob.glob(os.path.join(path_to_dir, '*.dat')):
    path, full_name = os.path.split(direction)
          # split filename and extension
    filename, _ = full_name.split('.')

          # read in record 
    record = wfdb.rdrecord(os.path.join(path, filename))

          # read in annotation
    annotation = wfdb.rdann(os.path.join(path, filename), 'atr')

        # Get the p_signal length and the annotations
    p_signal_length = len(record.p_signal[:,0]) # assuming we're interested in the first signal  
    chunk_len = record.fs * 10
        # Calculate the number of chunks
    num_chunks = p_signal_length // chunk_len
    df = pd.read_csv(os.path.join(path_to_csv, filename + '_segmentation.csv')).values

    for i in range(num_chunks):
      start = i * chunk_len
      end = (i + 1) * chunk_len
      if ast.literal_eval(df[i, 1]) != [start, end]:
        raise ValueError(f"An error occurred: {filename}, {i}. Expected: {[start, end]}. Actual: {ast.literal_eval(df[i, 1])}")
  

  print('pass the test')


#data_segmentation()
data_statistics('arrythmia_csv/DS2')
data_statistics('arrythmia_csv/DS1')
data_statistics('arrythmia_csv/DS3')
#file_check('arrythmia_csv')
#generate_all_symbol_set()


'''
annotation = wfdb.rdann(os.path.join('arrythmia_db', '102'), 'atr')
with open('annotation_aux.txt', 'w') as f:
     f.write(str([n for n in annotation.aux_note]))
     f.close()

with open('annotation_symbol.txt', 'w') as f:
     f.write(str([n for n in annotation.symbol]))
     f.close()
'''
