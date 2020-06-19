import sys

if len(sys.argv) != 3 :
  print('usage:')
  print('  ' + sys.argv[0] + ' input_fn output_fn')
  print('function:')
  print('  writes data for plotting num_sequences as a function')
  print('  of sequence length to "output_fn"; prints length')
  print('  of longest sequence to cout (add two for <bos>, <eos>)')
  print('delimiter:')
  print('  is hard-coded for comma\n')
  exit(9)

a = open(sys.argv[1])
a.readline() #discard header
out = open(sys.argv[2], 'w')

longest = 0
longest_seq = ''
longest_line_num = 0

data = {}
j = 0
for line in a :
  j += 1
  if j % 1000 == 0 : print(str(j/1000) + 'K lines processed')
  t = line.split(',')
  x = len(t[0])
  if x not in data :
    data[x] = 0
  data[x] += 1
  if x > longest : 
    longest = x
    longest_seq = t[0]
    longest_line_num = j-1

v = []
for ell in data :
  v.append( (ell, data[ell]) )
v.sort()


for d in v :
  out.write(str(d[0]) + ' ' + str(d[1]) + '\n')
print('\noutput written to: ', sys.argv[2] + '\n')
out.close()

print('\nlongest sequence length: ' + str(longest))
print('line number of longest: ' + str(longest_line_num))
print('longest sequence length: ' + longest_seq)
