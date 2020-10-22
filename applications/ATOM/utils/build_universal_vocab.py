#
# run with python 2.7
#
import string

a1 = string.letters
a2 = string.digits
a3 = string.punctuation
a4 = a1 + a2 + a3

out = open('vocab_universal.txt', 'w')
id = 0
for c in a4 :
  out.write(c + ' ' + str(id) + '\n')
  id += 1
out.write('<bos> ' + str(id) + '\n')
id += 1
out.write('<eos> ' + str(id) + '\n')
id += 1
out.write('<pad> ' + str(id) + '\n')
id += 1
out.write('<unk> ' + str(id) + '\n')
id += 1

out.close()
print('\nwrote file: vocab_universal.txt\n')
