Format for binary files; all values are floats

num_frames  #aka, num_samples
num_beads   #for now, will always be 184

#Repeating, for each frame:

  z-coordinates for each bead #184 entries

  #Repeating, for each beads in the current frame:
    euclidean distance between beads j and k, 
    j=0..num_beads-1, k=j+1..num_beads (16836 entries per frame)

Notes:
  Optional normalization of euclidean distances and/or Z-coordinates
  will be computed during the data_reader load method

