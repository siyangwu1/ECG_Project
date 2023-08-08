import wfdb
import numpy as np
import glob, os
import pandas as pd
import ast
path_to_dir = "atrial_fibrillation_db"
path_to_csv = "atrial_fibrillation_csv"
def generate_all_symbol_set():

     symbol_set = set()
     for direction in glob.glob(os.path.join(path_to_dir, '*.dat')):  
          path, full_name = os.path.split(direction)
          # split filename and extension
          filename, _ = full_name.split('.')

          # read in annotation
          annotation = wfdb.rdann(os.path.join(path, filename), 'atr')
          for ele in annotation.aux_note:

               symbol_set.add(ele)

     with open(os.path.join('atrial_fibrillation_csv', 'unique_symbol_in_AF_DB.txt'), 'w') as f:
          f.write(str([ele for ele in symbol_set]))
          f.close()          

def data_segmentation():
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
    aux_note = np.array(annotation.aux_note)
    sample = annotation.sample

        # calculate the length of each chunck
    chunk_len = record.fs * 10
        # Calculate the number of chunks
    num_chunks = p_signal_length // chunk_len

    label = []
    index = []
    sample = np.append(sample, float('inf'))
    aux_note = np.append(aux_note, aux_note[-1])
    for i in range(num_chunks):
      start = i * chunk_len
      end = start + chunk_len

      
      rhythm_begin = [sample[0], aux_note[0]]
      rhythm_end = []

      for tracker, (change_point, curr_rhythm) in enumerate(zip(sample[1:], aux_note[1:])):
        rhythm_end = [change_point, curr_rhythm]
            
        if (start < rhythm_begin[0] and end > rhythm_begin[0]):
          index.append([start, end])
          label.append('junk')
          rhythm_begin = rhythm_end
          break

        elif start < rhythm_begin[0] and end < rhythm_begin[0]:

          break

        elif start >= rhythm_begin[0] and end <= rhythm_end[0]:
          index.append([start, end])
          label.append(rhythm_begin[1])
          rhythm_begin = rhythm_end        

        else:
              
          rhythm_begin = rhythm_end
    # Create a DataFrame and export to CSV
    df = pd.DataFrame({
    'segment_idx': np.arange(len(label)),
    'segment_range': index,
    'classification': label
    })

    df.to_csv(f'atrial_fibrillation_csv/{filename}_segmentation.csv', index=False)        


def data_statistics(path_to_csv):

  junk_count = 0
  normal_count = 0
  AFIB_count = 0
  J_count = 0
  AFL_count = 0
  total_num = 0
  for direction in glob.glob(os.path.join(path_to_csv, '*.csv')):
    df = pd.read_csv(direction)
    label = df.values[:, -1]
    total_num = total_num + len(label)
    junk_count = junk_count + sum(label == 'junk')
    normal_count = normal_count + sum(label == '(N')
    AFIB_count = AFIB_count + sum(label == '(AFIB')
    J_count = J_count + sum(label == '(J')
    AFL_count = AFL_count + sum(label == '(AFL')


  
  with open(os.path.join(path_to_csv, f'{os.path.split(path_to_csv)[-1]}_data_stats.txt'), 'w') as f:

    f.write(f"total number of total segmentation is: {total_num} \n")
    f.write(f"total number of Normal pieces is : {normal_count} \n")
    f.write(f"total number of junk pieces is : {junk_count} \n")
    f.write(f"total number of AFIB pieces is : {AFIB_count} \n")
    f.write(f"total number of AFL pieces is: {AFL_count}\n")
    f.write(f"total number of J pieces is {J_count}")
    f.close()



def file_check():
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
#generate_all_symbol_set()
data_statistics(path_to_csv='atrial_fibrillation_csv')
data_statistics(path_to_csv='atrial_fibrillation_csv/DS1')
data_statistics(path_to_csv='atrial_fibrillation_csv/DS2')
data_statistics(path_to_csv='atrial_fibrillation_csv/DS3')
#annotation = wfdb.rdann("atrial_fibrillation_db/07859", 'atr')
#annotation
#file_check()