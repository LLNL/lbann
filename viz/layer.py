import pprint

class Layer :

  def __init__(self, a) :
    self._layer = self.__getLayer(a)
    self._parents = []
    self._children = []
    self._linked_layers = []
    self._attr = []
    for line in self._layer :
      if line.find('name:') != -1 :
        t = line.split()
        self._name = t[1][1:-1]
    for line in self._layer :
      if line.find('{') != -1 :
        t = line.split()
        self._type = t[0]
    for line in self._layer :
      if line.find('parents:') != -1 :
        t = line.replace('"', '')
        t = t.split()
        self._parents = t[1:]
    for line in self._layer :
      if line.find('children:') != -1 :
        t = line.replace('"', '')
        t = t.split()
        self._children = t[1:]
    for line in self._layer :
      if line.find('linked_layers:') != -1 :
        t = line.replace('"', '')
        t = t.split()
        self._linked_layers = t[1:]
    start =  0
    end = 0
    for j in range(1, len(self._layer)) :
      if self._layer[j].find('{') != -1 :
        start = j+1
      if self._layer[j].find('}') != -1 :
        end = j
        break
    self._attr = []
    for a in self._layer[start:end] :
      b = a.strip()
      if b.find('weight_initialization') != -1 :
        b = b.replace('weight_initialization', 'weight_init')
      self._attr.append(b)

  def __getLayer(self, a) :
    '''for internal use'''
    r = []
    n = 0
    for j in range(len(a)) :
      r.append(a[j][:-1])
      if a[j].find('}') != -1 :
        n += 1
        if n == 2 :
          break
    return r

  def name(self) :
    return self._name

  def parents(self) :
    return self._parents

  def children(self) :
    return self._children

  def type(self) :
    return self._type

  def linkedLayers(self) :
    return self._linked_layers

  def attributes(self) :
    return self._attr

  def printme(self) :
    pprint.pprint(self._layer)
