import sys

if len(sys.argv) != 3 :
  print(F'''
    usage: {sys.argv[0]} input_fn output_fn
    function: computes the length of each SMILES string
    output: each line of output contains a file name, followed by
            the length of each string
    where:
       "input_fn" contains the names of one or more smiles files;
       Assumes each file contains a single header line;
       Assumes delimiter is either tab or comma
    '''
  )
  exit(9)

a = open(sys.argv[1])
out = open(sys.argv[2], 'w')

sample_id = -1
num_files = -1
for line in a :
  out.write(line[:-1])
  print('opening: ' + line[:-1])
  b = open(line[:-1])
  num_files += 1
  b.readline() #discard header
  for line in b :
    sample_id += 1
    j = line.find(',')
    if j == -1 :
      j == line.find('\t')
    if j == -1 :
      print(f"failed to find delimiting character (comma or tab) on line # {sample_id} of file: {line[:-1]}")
      exit(9)
    out.write(' ' + str(len( line[:j] )))
  out.write('\n')
  b.close()
  if num_files == 3 : break

a.close()
out.close()
print(F'\noutput has been written to: {sys.argv[2]}\n')
