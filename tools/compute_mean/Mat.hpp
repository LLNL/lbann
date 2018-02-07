#ifndef _TOOLS_MAT_HPP_
#define _TOOLS_MAT_HPP_
#include <vector>
#include <cstddef> // size_t

template<typename T>
class ElMatLike {
 protected:
  int m_width;
  int m_height;

  std::vector<T> m_buf;

 public:
  ElMatLike() : m_width(0), m_height(0) {}

  int Width() const {
    return m_width;
  }

  int Height() const {
    return m_height;
  }

  void Resize(int w, int h);
  T *Buffer();
  const T *LockedBuffer() const;
};

template<typename T>
inline void ElMatLike<T>::Resize(int w, int h) {
  if ((w < 0) || (h < 0)) {
    m_width = 0;
    m_height = 0;
  }
  m_buf.resize(static_cast<size_t>(m_width*m_height));
}


template<typename T>
inline T *ElMatLike<T>::Buffer() {
  if (m_buf.size() != static_cast<size_t>(m_width*m_height)) {
    m_buf.resize(static_cast<size_t>(m_width*m_height));
  }
  if (m_buf.size() == 0u) {
    return nullptr;
  }
  return (&(m_buf[0]));
}

template<typename T>
inline const T *ElMatLike<T>::LockedBuffer() const {
  if (m_buf.size() == 0u) {
    return nullptr;
  }
  return (&(m_buf[0]));
}

using Mat = ElMatLike<lbann::DataType>;

#endif // _TOOLS_MAT_HPP_
