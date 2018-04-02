# coding=utf-8

source_1 = """#!/usr/bin/env bash
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx ids anywhere
# in the first comment block in the source file, i.e., in the first
# consecutive range of lines beginning with `#`.
#
# You can find out ReCodEx id on URL when watching ReCodEx profile.
# The id has the following format: 01234567-89ab-cdef-0123-456789abcdef.
#
# c6ef657e-2082-11e8-9de3-00505601122b (Anastasia Lebedeva)
# 08a323e8-21f3-11e8-9de3-00505601122b (Karel Ha)
#

INTERPRETER=/usr/bin/python3
SCRIPT=/home/mathemage/deep-learning-mff-uk-2018-summer-semester/labs/04/mnist_competition.py

# \"--batch_size\", default=50, type=int, help=\"Batch size.\")
# \"--dropout\", default=0.6, type=float, help=\"Dropout rate\")
# \"--cnn\", default=None, type=str, help=\"Description of the CNN architecture.\")
# \"--epochs\", default=10, type=int, help=\"Number of epochs.\")
# \"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
ARGS=(
#\"--cnn=CB-10-3-2-same,M-3-2,F,R-100\"
#\"--cnn=CB-10-3-2-same,M-3-2,CB-10-3-2-same,M-3-2,F,R-100\"
#\"--cnn=CB-10-3-2-same,CB-10-3-2-same,M-3-2,F,R-100\"
#\"--cnn=CB-10-3-2-same,CB-30-3-2-same,M-3-2,F,R-100\"
#\"--cnn=CB-10-3-2-same,F,R-100\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=64\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=256\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=1024\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=2048\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=256  --epochs 30\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=1024 --epochs 30\"
#\"--cnn=CB-10-3-2-same,F,R-100 --batch_size=2048 --epochs 30\"
#\"--cnn=CB-20-3-2-same,M-3-2,F,R-300 --epochs 30 --batch_size 64\"
#\"--cnn=CB-20-3-2-same,M-3-2,F,R-300 --epochs 30 --batch_size 256\"
#\"--cnn=CB-20-3-2-same,F,R-300       --epochs 30 --batch_size 64\"
#\"--cnn=CB-20-3-2-same,F,R-300       --epochs 30 --batch_size 256\"
#\"--cnn=CB-20-3-2-same,F,R-300       --epochs 120 --batch_size 256\"
#\"--cnn=CB-10-3-1-same,CB-10-3-1-same,F,R-300 --epochs 30 --batch_size  50\"
#\"--cnn=CB-10-3-1-same,CB-10-3-1-same,F,R-300 --epochs 30 --batch_size 256\"
#\"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs 30 --batch_size  50\"
\"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs 30 --batch_size 256\"                          # <- best so far
#\"--cnn=CB-20-3-2-same,M-3-2,F,R-300 --epochs 100 --batch_size 64 --learning_rate 0.01 --learning_rate_final 0.005\"
#\"--cnn=CB-20-3-2-same,F,R-300                --epochs 120 --batch_size 256\"
#\"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs 120 --batch_size 256\"     # <- best at epoch #78.01 -> acc 99.94
\"--cnn=CB-20-3-1-same,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-20-3-1-same,M-2-1,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-20-3-1-same,M-3-2,CB-20-3-1-same,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-10-3-1-same,M-2-1,CB-10-3-1-same,M-2-1,CB-10-3-1-same,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,M-2-1,F,R-300 --epochs  78 --batch_size 256\"
\"--cnn=CB-10-3-1-same,CB-10-3-1-same,CB-10-3-1-same,M-3-2,F,R-300 --epochs  78 --batch_size 256\"
)

for configuration in \"${ARGS[@]}\"; do
    command=\"$INTERPRETER $SCRIPT $configuration\"
    for i in $(seq 1); do
        echo ${command}
        ${command}
        echo
    done
done
"""

source_2 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


# switch off GPU: CUDA_VISIBLE_DEVICES = \" \" python3


# noinspection SpellCheckingInspection
class Network:
	WIDTH = 28
	HEIGHT = 28
	LABELS = 10

	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                             intra_op_parallelism_threads=threads))

	def construct(self, args, batches_per_epoch, decay_rate):
		with self.session.graph.as_default():
			# Inputs
			self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
			self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
			self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

			# Computation
			latest_layer = self.images
			# Add layers described in the args.cnn. Layers are separated by a comma and can be:
			cnn_desc = args.cnn.split(',')
			depth = len(cnn_desc)
			for l in range(depth):
				layer_name = \"layer{}-{}\".format(l, cnn_desc[l])
				specs = cnn_desc[l].split('-')
				if specs[0] == 'M':
					# - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
					latest_layer = tf.layers.max_pooling2d(inputs=latest_layer, pool_size=int(specs[1]), strides=int(specs[2]),
					                                       name=layer_name)
				if specs[0] == 'F':
					# - F: Flatten inputs
					latest_layer = tf.layers.flatten(inputs=latest_layer, name=layer_name)
				if specs[0] == 'R':
					# - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
					latest_layer = tf.layers.dense(inputs=latest_layer, units=int(specs[1]), activation=tf.nn.relu,
					                               name=layer_name)
				if specs[0] == 'CB':
					# - CB-filters-kernel_size-stride-padding: Add a convolutional layer with BatchNorm
					#   and ReLU activation and specified number of filters, kernel size, stride and padding.
					#   Example: CB-10-3-1-same
					# To correctly implement BatchNorm:
					# - The convolutional layer should not use any activation and no biases.
					conv_layer = tf.layers.conv2d(inputs=latest_layer, filters=int(specs[1]), kernel_size=int(specs[2]),
					                              strides=int(specs[3]), padding=specs[4], activation=None, use_bias=False)
					# - The output of the convolutional layer is passed to batch_normalization layer, which
					#   should specify `training=True` during training and `training=False` during inference.
					batchnorm_layer = tf.layers.batch_normalization(inputs=conv_layer, training=self.is_training)
					# - The output of the batch_normalization layer is passed through tf.nn.relu.
					latest_layer = tf.nn.relu(batchnorm_layer, name=layer_name)

				# # Implement dropout on the hidden layer using tf.layers.dropout,
				# # with using dropout date of args.dropout. The dropout must be active only
				# # during training -- use `self.is_training` placeholder to control the
				# # `training` argument of tf.layers.dropout. Store the result to `hidden_layer_dropout`.
				# hidden_layer_dropout = tf.layers.dropout(hidden_layer, rate=args.dropout, training=self.is_training,
				# 																				 name=\"hidden_layer_dropout\")
				# # output_layer = tf.layers.dense(hidden_layer_dropout, self.LABELS, activation=None, name=\"output_layer\")

			# Store result in `features`.
			features = latest_layer

			output_layer = tf.layers.dense(features, self.LABELS, activation=None, name=\"output_layer\")
			self.predictions = tf.argmax(output_layer, axis=1)

			# Training
			loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
			global_step = tf.train.create_global_step()
			learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch, decay_rate,
			                                           staircase=True)
			# - You need to update the moving averages of mean and variance in the batch normalization
			#   layer during each training batch. Such update operations can be obtained using
			#   `tf.get_collection(tf.GraphKeys.UPDATE_OPS)` and utilized either directly in `session.run`,
			#   or (preferably) attached to `self.train` using `tf.control_dependencies`.
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=\"training\")

			# Summaries
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
			summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
				self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
				                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in [\"dev\", \"test\"]:
					self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
					                           tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, images, labels):
		self.session.run([self.training, self.summaries[\"train\"]], {self.images: images, self.labels: labels,
																																self.is_training: True})

	def evaluate(self, dataset, images, labels):
		accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels,
		                                                                          self.is_training: False})
		return accuracy

	def predict(self, images):
		return self.session.run(self.predictions, {self.images: images, self.is_training: False})


if __name__ == \"__main__\":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
	parser.add_argument(\"--dropout\", default=0.6, type=float, help=\"Dropout rate.\")
	parser.add_argument(\"--cnn\", default=None, type=str, help=\"Description of the CNN architecture.\")
	parser.add_argument(\"--epochs\", default=10, type=int, help=\"Number of epochs.\")
	parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
	parser.add_argument(\"--learning_rate\", default=0.001, type=float, help=\"Initial learning rate.\")
	parser.add_argument(\"--learning_rate_final\", default=None, type=float, help=\"Final learning rate.\")

	args = parser.parse_args()

	# Create logdir name
	args.logdir = \"logs/{}-{}-{}\".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
		\",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists(\"logs\"): os.mkdir(\"logs\")  # TF 1.6 will do this by itself

	# Load the data
	from tensorflow.examples.tutorials import mnist

	mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42,
	                                        source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

	# Construct the network
	# set up decay rate
	if args.learning_rate_final is not None:
		decay_rate = np.power(args.learning_rate_final / args.learning_rate, 1 / (args.epochs - 1))
	else:
		decay_rate = 1.0
	batches_per_epoch = mnist.train.num_examples // args.batch_size

	network = Network(threads=args.threads)
	network.construct(args, batches_per_epoch, decay_rate)

	# Train
	for i in range(args.epochs):
		j = 0
		while mnist.train.epochs_completed == i:
			print(\"Epoch #{} \\t Batch #{}\".format(i, j))
			j += 1
			images, labels = mnist.train.next_batch(args.batch_size)
			network.train(images, labels)

		network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)

	# Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
	test_labels = network.predict(mnist.test.images)
	test_filename = \"{}/mnist_competition_test.txt\".format(args.logdir)
	with open(test_filename, \"w\") as test_file:
		for label in test_labels:
			print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;7%VCRb2oX0)oax6*GjyvYTC`s}Z?E4>dgkac{%n=7fv}Z9C2yLvSj;GvI6~#Qz^B=K$n%0+Z_d+C_KFzILwpcdH>kw|Ezv+<XwK;+u~kCIG8b*anJS!qYZ}GW=YDQj@#}md3BWGf%*tbs;|bI=GjWo98HwhN7~}kJjB{<Rzv?<Vpw%MHO*J*QxzWYy<X^((dmK|L7EEowmLX#hse}m792jYV1(765r@m<Q!LGTIMHjrGe6Kn0YIJio3*BF|x58+pE#s@R+wnM2N`-*5)JXIyrm>d=}cJ^7EBjZ42$#s(^siEn9)*Z9r6!IksVtx>h73(6x29wUc#epuYaAIVm*k&uOZ;5)I>RdU(iazpBtVxGki9+aamX?v&%E2^}+<Z&Q1MaIYFJERYXj;!X9h_0r2Y&HoVKebHW`%}NUOje#{<0pRHx5u{?1q)UoC^*)(_&R5+AJ8(AKSW^*TqXL?8z6_k@gc(l<?<3W*^^<vIJFAg)X1Wkfw_<St3nCl6kkMDCKrZTKh=4;1oA;y7Zb~jDQIQ?HtU$)`RtTx}u8Bk$c=uFxMr~K67F$7l=2e?_>Y%}t;vlngqM~R_rHG}K0`AIy8pm((Xp<34)<>nlhBkF<P0K5?T>+4{j-?dMKm#rL5O@%g@1`gFJ%)b8QJL5B;*Q}?1H}`AIdnao&a0W>5GP=kJaCDxVKO<%z%u$vwNXKO0;a>e5_J{n&Z#7p7H%(FY@>}}<GTFp<}pk4DG3V4@%Xa>Nn-E{r}^XOL^fsRJPFa(fj!ArP22B~0W+6Yl`9~i6+Mg80nCab`*0P(l}2hw>x(EhvSePs3W+i;4<rUUDPq@M>j#p~3}VNdrh4i&)~r?2{a4VaV6QLb<KTb%J3?goeGLaLyLhgR4~w$+=N<0QJ;m(o8l<l!zHK#D5oyja>AF0z*>4%cEPKaySW*#;95?bpw$oaiuWW~M2Pl_ABqj1Sw3yO=)AUkJ2dQyK*b0E{YfMR-B569Zl`2nrYC}n)F%#DH7k?CZ`N^NKT$9B)@=fx>@obrNIXMKLZltQZkpWLxvf(XOgh<H1NeH@486IsK<4nSn*Tz&=5G_{5w|4BhE<$)3@{!%*2FVPhk3MVR$|s;d!e?QJkA(=mx?SHtRL6+UF{)Y<-yF}}=>j7C1na@-3@?Sy2;UnbGKYln%95FP0dl>*Dr9T*l9*3ihanF^RNcI+I}#>Zj}1CTJFL1i3@xZBh#QK~77TFa${W`O%)zqT+F*#BFpVQijL{U!z_}`M$qLagvrkiMRu&rLYeCu2NlQT%&<GAPG}pPj%xYe9**|Ud7wba|ltG@)S%%|G0U=USWM=->IzQ($H$KLDp=<i?cp-s|Cjamnqw}bDzeY<%WdMfUlWy|WH6qn~4u6Lycmim5P57E@!#6ZM+I^+ku+vM%_`{!++P#HP)LQ6%YYFP&ou`fJhc0j`ALE<DsLq%;Hr(UrQI2*ExB^J+nTG>eu;cZTmgx33`kHdpnqQqXR}I=}ksDd4r1Q~@V|yuwqNOe`SrM#fX=eP<YLo1B`1Eb~weEWRxlsshC&wEQ=12<gOhW?}z?N*4@=AO?U~ff7`OIM3HsJQu$jN#y<C!=+5;#YHXY_vQEg%9m5)K=j*-&4zQ^dF;hoD<EX9MrN`~6~YuWYVCeY;Tnn|X_vs)eugL1YC>lx3y~=qD=G)kmj_!*R3%Mz>Y%0Hmjo%k2`ux9^#=!~_$j6Qk9D585yLkoEkunT~BcJ~o<`%%(4Zpv)Z|QDOdrWSM4c)a_lwnWZaO{#r+I7Ufo#K+=sAT;0U@mVJ|%A<%nJak`xQfrq7dEZDr7c|Y`l1|bDJE>Lh0h_j)3vvkd^;nFx;(?G{WOgye2I~K!IHC4mo7grxVw<9!D;)O$Y-W5b;mDD5ANaNc0;WvJ24ZhdxsIj^@7nDj5AeUTNT}$G<EP<3g649!1_?E(=)1T<#>D9mrs?n7ML{aehc9Ehd!#IbsUJB-zKI>$ycLJkSrv5mKKDzf<qw&J_38Y*rwhmgdILLE9fT#`bR`OrAiOfIHH4WPMyg+YDu$xU!IE@+{NySM`dmRD?W1FdP$nJeuhqO%*%y~Sps-mm<)eQtbk`vfynz5xjHI?^+5l`;Uh^eI>&DUO?H55;Uf^TVRF%13+4J{uUebe%Ba-p9lV6%H3O0H0D>~xq;vTBvKwvGdYi0zyOL&B@xmq}UCz<n)}+n-bAm6V+Z)f#k}u%Vz2)&!wZSl8pQomqD>-&7KfA5U^{f=TU_)_jnoF^~64-%IG?|8!T^B!)#FtStErGa>upzaPRx&DIU$f!YeTposA*u`Nx;eAkv!?>Qk98<mr|-@C#z9wu(X+3!vLJ<A*a<U+zZrT|zs#3-qQsP$-a+n6yA5(ex-+@pTONVDugdO#(S9^eqhH1*waYsJ8QHlxL;u0e<L+hisN@*%WD72D>n^#&3S)rFDpID*ZuqNe`6$I2qtsV*XTrH38AHS9ILD*UQNgh01v=lQXuPaMAS(jx6}biLBQmas~cU%6)ADu>6yG^Z^~vTat+>GBf6N8&p`+F#)t!9>aOQrvO>ElArwE0*rPQ{*OjN;S2nZ`nEZ4k7uNKEN@}!q43lGH9TQwp27uSoL+-$@*P6RuyT<Az%NCw#cv<r}w<<JbIpLFN?pdE?lZmesE1Yn9m$X%p<P;$T~B*sx!R}p&R3zG78zO{DMVJHwYD|AJa&Q$Pbf~SZBo2jm{2#)J@xUEmBwuWKCU=HQ}`^iNV`vRMIHCLw(GH3I|>q%c%dMstSf>Z!`|uZq$EbB`SX&g6qp}c26MQmhRQ79E12ND68y$tZqZ@i^e4)`GYoSpiA67zS~e0gpb_;I(S{;{z-sfST?}3eGj%ztu+;UEf^h-e#Nx2-v0;s<mrgyRHS5h4VH<D|K518T+=K#9Q{0Lb;q;<Ug_-+eJlR`$S4ex{vZ1#?W#O?RldmqukJ6GPlqv*r%suV8dBm7y!_7oAiT@U-+3f{F?_q${Dj6Y#mnX88$|B@)nHxo;*@YD{vDORWjads|8lm+d#`yoBzQ0O*&C9yM&zj;8S%Re(I|bWjS@f_&IsJDvGqrZ$O&g4mj+wWx{CcJWFg)?Rf)P-$RSc+nh8JH@^|!#4QB}_VIF`0*+L5=+_M7D+=DdtYe53DJmYHBRjlpa&4A9jZ=Kbe_R8|)OyAt?Tq#kSF?JHm_&UV2=JEQ}clKe~U5`6B%#*m3-p9h&-N}%iZJXy;)WI#Pl|I$8M5z@X{UqS*p_&b;ZkXQaV^s4Td9}M)Lv@^hf)HWP(+<&j1L#nx+FqokWL<KoXK<a2%<MVgJQNW0u{AUb3sJHmk7Uj$CRH_axLdEl5}bumN$#aoK}D?+b`eaHoA@8p^P1d*@SJ_E|LbxS9LpacH7^9LbofhkMu6+$mAYkiwzc8j%oYx>W%tDS3?Xwy9hbOBD|;poAfYJl@V^*Ayms;d2<xx|99V<S?5Y*3-GAjd_lE#LDnrv3yJRv#88I^|CEB$r_7_#d0d7ZbwJ-HD&Iz+c>8!DqbDGKo7eoBrmlkxOpIfkH%$|A&AI6?(f>qep3<v!S2f`s+%%1tHx8ex;bV;hUNyDQk_8(T#VO2+}ZIEu$8iZCaAu%K-Vc1d*OktEEyy86mxJAtPse_<GY%iD&V;hsxHWyp2c*t{)n{O0d&!`v#ReSv%d1igQgnondPvv0pC;!>Zg<)6LNtg<eDGH6r$&RA*=>e$qq22Q0S1iLCTeL%>#bZJ<F&0%+EATjZoh)r{)fehhgPe~X>0ax5s<Oatf1g8{awIipdC*+s9-&^^J5W$brF^er%ku}m5aT&N|4=Wvx2W#AOZsp-Qd46xnV?<HKS6ZE`E}kmNwe=9f+0B4TQ`ztTT5&7BqU1e<CWhHzCZ~V&;@dv^X<;5;Q{j{sGRX~pb#oV+R=X@_L8<Re^=}353~xeIGfN>9890oX#+nRXg%t>S1BAs(Y0K}Fi0M(Lz&8f$kE<(t_dQ#mp$XUJ+F9MRm!kMlfdDm7rC#3n#85Jq^g3GEY8a$dh4%-7|*nD`evc7cD)vYHMN%hhzjg}4$5W-2h+VkYwquxELgUTJM+s*`a4#LaCQv7u!}M=1yFQ(=o%!|;2UTlMkEq$l{O%U=fJb5IHWJix|t)Jb1k%)6#)5sWwoQUFJ;W8%&doD%$;^&rB;!a{@FT)fy(ZC`!!NFLaBN2>2LW&Az{^43irt-OcXd=@I5DvE2rWrZ85f(0(Sx=f1H4rTiL_!M1jl!>xKYe0#eRk%&J8b(iz!OiN-otD?TPo-^T$`yr=?_UZa;hxQP^GaeGjj(iDPE&+-on5X%tS_dZ9q-aH3#t;+m;(AH#Y>WkF`kJnNl(1vPr^I?kq%8@uN#bxlu+U&5@vsipG22Dv6XML}=9H_AMf^%*5z^+zv4ysC~T!`_I;DRz?7%m<Z5%9zySwqfa@j)~}zMlt1yVig7MC)&f-nyuy2Txf!FKJwY9bgxBKRhl2c(~>l`Ri}EOB6sbWu#Bi9ex+(KTfyah#Rs^tZAno&B^WgtSWYXF_zL>32b%Z;(PvwSuU8D@@WKwRPR#zw&;QA1XoD#iy1-GtRk2dH$=o_iJkLh@+;3X^@5=N5Lg{F6W@+7Qz=%?*pFR^F`NY4I14G{u7bk=f{Xtgd1E#%5VP}f_j@#4O7)t3Sf7T{WL3hYmxw}*hL(ORNiuco_~;j7lq<wplNHHMo=>63PxC&GZP(w{bEXQZ-mvxrO%JAtg0`ZVq`qsna&J1|%bd@RW_7gxJ3BNn?`2$Assflqg-+wrZX5)OV9x@~12qT7!_R#@xQn9#t{WRbhn5}`Z&BMGGH!F}RAGi&m^3;dgVE67qC>&|Ibw>kURO$DJn+bP^$>J8GYvlS=s_@NEnCahmW|<~j>MB+FA{z3BNsg>0zZ4kD;j9{opcvn4pulXEFq|$t@EQ}TAwoeb~E11=H1c8lsM=&^?;BJdHE+$sux~i{7PW{sP5_(5QnWTqD0$agonF=D>3M%y*R8kJbuF!-T>6IALQu}dQ9p;R2FhbS0+kyk-^bh^s3sD(6nHJp-GLY!`u<yMxww!BH(PqbCx}vNS+R^le9atLeV494<vx=bnMfJvGO-gq6!Mg?FHaopOZs=3zxmh1A(iPu_GM0baQD<CH@hB5W;aACcH-l-g-Bn9dT$E&?wb<PN7%2V!%qK6jt$_qg*1gN+WCaPiUA&`{a@w$#-h&Tr8~p6>}nhkWicZVvK&KJ=1Cep9N>^CFHKqXi(})h=xVKSi{M~>rPptwO&BJV6(r!eq`MTN!-Y@nS4*PUx_JD@{c5eUXt3VA6qb#6=@Sq2peQ!Y7|y&2^=w*>dC^2+8326EVau!0FHxGM_iLfBB8TWLKzkwh2|X8{mREb&;~~FQm9pa(DNUKjB_Ju2z_BO7(hW!c43UD_(LuC(kCYh%&p$3%D*DZfs;S&`3d>b);iN#2%5`Kp>q<qEIoju3||%IL&-<8@#5F)?Cdg$7NZVN$99izKeE+sOhgNTLk$#h0A3g#@*EWIs&bU{FJ@ioC!`uowHO$CwuETbT1*!Af+sr`4;CfSv{1T(^g<3-9R{vV7VleAlb{?2l|zP6xvR<YZ(vOT#x`zB`zBmwyk>Z#3QhgH|BK6DBJnD7`NkCUn?wL@y=vE{dNk4U_pwH*a4M}eWYvLskxl-)uwR3>y+r&!W5<|6v3qunAAR)s6i8?2+AX!;WC$8G4s~1qMgkmg;A!E)JIDbD<Ehq8@efEfwwn;DULce+rF?7}Znb;pw-z4>A52KZ7;6DE>Na25{;&}xbBNi>B2&cyGOcKjECb9oNH*MULi$n|mL_%E*@I!VlYnluKXbOIeP1v%PX^BYk=7;^FBptwE~;3hRDcoWy3b4bobv*g#^LQm4#8I^acIr`idCHCpGE&6A#u!vfW}O%^6^*Ts5}IQcwZ>l=-6jp<GcFOI;Ep2mM=_GxdEU`Hcam!Z+vf_ddw4TfGfH6GGbUxbc#Fn60s>AaW4Q>u+KY&qf}~D8spu=zlS5XzI}QDzzyf(xJ^96Tf~74Uul<&fvVoNRle`x&W`|ByZf3L7~DshK=X)9uE*wOJunVQi-D&u4Q?H1n9nZgg#}GT?i1JJ!tKMEdJB0je6v;oK;4>=pF`1Ef1|D>Oj*Zz8nj;2?0=v84AX+`P8H)FGNaplNL~y|qzzPtkoHn>+wl<@8xGiXFsMo$k;C&Q-Kup^$9rrY-Nya&rR$J5yTO)~%d>zA3}KCuWCP@^>qj^(!ZH&^y*gLo!ZjPX`_^K7Cg0Gy00000<ZXanE27F{00HqPpqv2!6|H0QvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
