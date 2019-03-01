The following cmds were run in the LBANN root directory.

```shell
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt brief=1 ranksep=.7 output=rnn_1
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt brief=1 ranksep=.7 output=rnn_1 output=jpg
```

* Output: `rnn_1.pdf`, `rnn_1.jpg`
* Notes:
  * linked layers are enclosed by dotted rectangles
  * `ranksep=.7` increases readability (IMO)

```shell
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt brief=1 output=rnn_1a
```

* Output: `rnn_1a.pdf`
* Notes: didn't specify `nodesep=.7`; harder to interpret (IMO)

```shell
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt ranksep=.7 output=rnn_2
```

* Output: `rnn_2.pdf`
* Notes: same as above, but print layer names as well as types

```shell
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt full=1 ranksep=.7 output=rnn_3
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext prop=scripts/proto/scripts/viz/properties_rect.txt full=1 ranksep=.7 output=rnn_3 format=jpg
```

* Output: `rnn_3.pdf`, `rnn_3.jpg`
* Notes: `full=1` prints all layer attributes

```shell
$ lbviz model_zoo/models/char_rnn/model_char_rnn.prototext ranksep=.7 output=rnn_4
```

* Output: `rnn_3.pdf`
* Notes:  didn't specify properties file, so uses the default `properties.txt`
