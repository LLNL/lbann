
int s = socket(AF_INET, SOCK_STREAM);

sockaddr_in addr;
addr.sin_port = htons(2000);
addr.sin_addr = IN_ADDR_ANY;

bind(s, &addr, sizeof(addr));

int efd = epoll_create1();

fctnl(s, F_SETFL, SO_NONBLOCK)

struct epoll_event event;

event.data.fd = s;
event.events = EPOLLIN | EPOLLOUT | EPOLLHUP;
epoll_ctl(efd, EPOLL_CTL_ADD, s, &event);

struct event *events = calloc(MAXEVENTS, sizeof(event));
while (true) {
  int n = epoll_wait(efd, events, MAXEVENTS, -1);
  for (int i = 0; i < n; i++) {
    if (events[i].events & EPOLLERR)) {
      //error
    } else if (events[i].events & EPOLLIN) {
      //read
    } else if (events[i].events & EPOLLOUT) {
      //write
    }
  }
}

---


while (true) {
  int cfd = accept(lfd, ...);
  if (cfd >= 0) {
    fctnl(s, F_SETFL, SO_NONBLOCK);
    struct epoll_event event;
    event.events = EPOLLONESHOT | EPOLLIN;
    event.data.ptr = client_new(cfd);
    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &event);
  }
}

while (true) {
  int i;
  int n = epoll_wait(epfd, events, MAXEVENTS, timeout);
  for (int i = 0; i < n; i++) {
    struct client *client = events[i].data.ptr;
    enum next_action next = client_read_write(client);
    int want = 0;
    switch (next) {
      case NEXT_RDONLY: want=EPOLLIN; break;
      case NEXT_WRONLY: want=EPOLLOUT; break;
      case NEXT_RDWR: want=EPOLLOUT|EPOLLIN; break;
      case NEXT_CLOSE: close(client->fd); break;
    }
    if (want) {
      events[i].events = want | EPOLLONESHOT;
      epoll_ctl(epfd, EPOLL_CTL_MOD, client->fd, &events[i]);
    }
  }
}
