import sys

if len(sys.argv) != 3 :
  print(F'''
    usage: {sys.argv[0]} input_filename output_filename
    where:
      "input_filename" is a SMILES csv filename
    function:
      computes the frequency of each character in the vocabulary,
      and prints same to "output_filename"
  '''
  )
  exit(9)

out = open(sys.argv[2], 'w')

a = open(sys.argv[1])
a.readline() # discard header
h = {}
j = 1
for line in a :
  k1 = line.find(',')
  if k1 == -1 :
    k1 = line.find('\t')
    if k1 == -1 :
      print('failed to find comma or tab delimiter on line # ' + str(j))
      exit(9)
  s = line[:k1]
  for c in s :
    if c not in h :
      h[c] = 0
    h[c] += 1  
  j += 1
  if j % 1000 == 0 : print(str(j/1000) + 'K samples processed')

v = []
for c in h.keys() :
  v.append( (h[c], c) )
v.sort()

for x in v :
  print(x)
  out.write(str(x[0]) + ' ' + str(x[1]) + '\n')
out.close()
print('\n\nOutput has also been written to: ' + sys.argv[2] + '\n')
