import pprint

class properties :
  def __init__(self, fn) :
    a = open(fn).readlines()
    shapes = {}
    colors = {}
    arrows = {}
    self._layers = {}
    for j in range(len(a)) :
      if a[j].find('shapes_and_colors') != -1 :
        k = j+1
        while len(a[k]) > 3 :
          t = a[k].split()
          shapes[t[0]] = t[1]
          colors[t[0]] = t[2]
          arrows[t[0]] = t[3]
          k += 1
    for j in range(len(a)) :
      if a[j].find('layer_names_and_overrides') != -1 :
        k = j+1
        while len(a[k]) > 3 :
          t = a[k].split()
          layer_type = t[0]
          layer_name = t[1]
          self._layers[layer_name] = [shapes[layer_type], colors[layer_type], arrows[layer_type]]
          if len(t) > 2 :
            for i in t[2:] :
              i = i.strip()
              t2 = i.split('=')
              if t2[0] == 'shape' : self._layers[layer_name][0] = t2[1]
              if t2[0] == 'color' : self._layers[layer_name][1] = t2[1]
              if t2[0] == 'arrow' : self._layers[layer_name][2] = t2[1]
          k += 1

  def shape(self, name) :
    if not self._layers.has_key(name) :
      print 'Nothing known about this layer:', name
      print 'Please check your properties file'
      print
      exit(0)
    return self._layers[name][0]

  def color(self, name) :
    if not self._layers.has_key(name) :
      print 'Nothing known about this layer:', name
      print 'Please check your properties file'
      print
      exit(0)
    return self._layers[name][1]

  def arrow(self, name) :
    if not self._layers.has_key(name) :
      print 'Nothing known about this layer:', name
      print 'Please check your properties file'
      print
      exit(0)
    return self._layers[name][2]
