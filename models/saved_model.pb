 ы
Ю	 	
B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108Р

while/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*(
shared_namewhile/gru_cell_1/kernel

+while/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpwhile/gru_cell_1/kernel*
_output_shapes

:`*
dtype0

!while/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*2
shared_name#!while/gru_cell_1/recurrent_kernel

5while/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!while/gru_cell_1/recurrent_kernel*
_output_shapes

: `*
dtype0

while/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*&
shared_namewhile/gru_cell_1/bias

)while/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpwhile/gru_cell_1/bias*
_output_shapes

:`*
dtype0

while/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namewhile/dense_2/kernel
}
(while/dense_2/kernel/Read/ReadVariableOpReadVariableOpwhile/dense_2/kernel*
_output_shapes

: *
dtype0
|
while/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namewhile/dense_2/bias
u
&while/dense_2/bias/Read/ReadVariableOpReadVariableOpwhile/dense_2/bias*
_output_shapes
:*
dtype0

while/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namewhile/dense_3/kernel
}
(while/dense_3/kernel/Read/ReadVariableOpReadVariableOpwhile/dense_3/kernel*
_output_shapes

: *
dtype0
|
while/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namewhile/dense_3/bias
u
&while/dense_3/bias/Read/ReadVariableOpReadVariableOpwhile/dense_3/bias*
_output_shapes
:*
dtype0

while/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namewhile/dense_4/kernel
}
(while/dense_4/kernel/Read/ReadVariableOpReadVariableOpwhile/dense_4/kernel*
_output_shapes

: *
dtype0
|
while/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namewhile/dense_4/bias
u
&while/dense_4/bias/Read/ReadVariableOpReadVariableOpwhile/dense_4/bias*
_output_shapes
:*
dtype0

while/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_namewhile/dense_5/kernel
}
(while/dense_5/kernel/Read/ReadVariableOpReadVariableOpwhile/dense_5/kernel*
_output_shapes

: *
dtype0
|
while/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namewhile/dense_5/bias
u
&while/dense_5/bias/Read/ReadVariableOpReadVariableOpwhile/dense_5/bias*
_output_shapes
:*
dtype0
p

rnn/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*
shared_name
rnn/kernel
i
rnn/kernel/Read/ReadVariableOpReadVariableOp
rnn/kernel*
_output_shapes

:`*
dtype0

rnn/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*%
shared_namernn/recurrent_kernel
}
(rnn/recurrent_kernel/Read/ReadVariableOpReadVariableOprnn/recurrent_kernel*
_output_shapes

: `*
dtype0
l
rnn/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*
shared_name
rnn/bias
e
rnn/bias/Read/ReadVariableOpReadVariableOprnn/bias*
_output_shapes

:`*
dtype0

NoOpNoOp
Э+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*+
valueў*Bћ* Bє*

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
 	keras_api
~

!kernel
"recurrent_kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
h

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
a
@	constants
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
a
E	constants
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
f
J0
K1
L2
!3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
 
f
J0
K1
L2
!3
"4
#5
(6
)7
.8
/9
410
511
:12
;13

Mlayer_regularization_losses
Nnon_trainable_variables
Ometrics
	variables
regularization_losses

Players
trainable_variables
 
 
 
 
 

Qlayer_regularization_losses
Rnon_trainable_variables
Smetrics
	variables
regularization_losses

Tlayers
trainable_variables
 
 
 
 

Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
	variables
regularization_losses

Xlayers
trainable_variables
~

Jkernel
Krecurrent_kernel
Lbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
 

J0
K1
L2
 

J0
K1
L2

]layer_regularization_losses
^non_trainable_variables
_metrics
	variables
regularization_losses

`layers
trainable_variables
ca
VARIABLE_VALUEwhile/gru_cell_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!while/gru_cell_1/recurrent_kernel@layer_with_weights-1/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEwhile/gru_cell_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
 

!0
"1
#2

alayer_regularization_losses
bnon_trainable_variables
cmetrics
$	variables
%regularization_losses

dlayers
&trainable_variables
`^
VARIABLE_VALUEwhile/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwhile/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1

elayer_regularization_losses
fnon_trainable_variables
gmetrics
*	variables
+regularization_losses

hlayers
,trainable_variables
`^
VARIABLE_VALUEwhile/dense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwhile/dense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1

ilayer_regularization_losses
jnon_trainable_variables
kmetrics
0	variables
1regularization_losses

llayers
2trainable_variables
`^
VARIABLE_VALUEwhile/dense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwhile/dense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51

mlayer_regularization_losses
nnon_trainable_variables
ometrics
6	variables
7regularization_losses

players
8trainable_variables
`^
VARIABLE_VALUEwhile/dense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwhile/dense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1

qlayer_regularization_losses
rnon_trainable_variables
smetrics
<	variables
=regularization_losses

tlayers
>trainable_variables
 
 
 
 

ulayer_regularization_losses
vnon_trainable_variables
wmetrics
A	variables
Bregularization_losses

xlayers
Ctrainable_variables
 
 
 
 

ylayer_regularization_losses
znon_trainable_variables
{metrics
F	variables
Gregularization_losses

|layers
Htrainable_variables
FD
VARIABLE_VALUE
rnn/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUErnn/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUErnn/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 

J0
K1
L2
 

J0
K1
L2

}layer_regularization_losses
~non_trainable_variables
metrics
Y	variables
Zregularization_losses
layers
[trainable_variables
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1rnn/bias
rnn/kernelrnn/recurrent_kernelwhile/gru_cell_1/biaswhile/gru_cell_1/kernel!while/gru_cell_1/recurrent_kernelwhile/dense_2/kernelwhile/dense_2/biaswhile/dense_3/kernelwhile/dense_3/biaswhile/dense_4/kernelwhile/dense_4/biaswhile/dense_5/kernelwhile/dense_5/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_200126
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cf74b9b68c134f939766d637bb13d6e3/part
m

StringJoin
StringJoinsaver_filenameStringJoin/inputs_1"/device:CPU:0*
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Ь
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ѕ
valueыBшB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUE

SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 
Л
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices+while/gru_cell_1/kernel/Read/ReadVariableOp5while/gru_cell_1/recurrent_kernel/Read/ReadVariableOp)while/gru_cell_1/bias/Read/ReadVariableOp(while/dense_2/kernel/Read/ReadVariableOp&while/dense_2/bias/Read/ReadVariableOp(while/dense_3/kernel/Read/ReadVariableOp&while/dense_3/bias/Read/ReadVariableOp(while/dense_4/kernel/Read/ReadVariableOp&while/dense_4/bias/Read/ReadVariableOp(while/dense_5/kernel/Read/ReadVariableOp&while/dense_5/bias/Read/ReadVariableOprnn/kernel/Read/ReadVariableOp(rnn/recurrent_kernel/Read/ReadVariableOprnn/bias/Read/ReadVariableOp"/device:CPU:0*
dtypes
2
h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 

SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
~
SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtypes
2
Ѓ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
Я
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ѕ
valueыBшB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUE

RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 
е
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2
D

Identity_1Identity	RestoreV2*
T0*
_output_shapes
:
V
AssignVariableOpAssignVariableOpwhile/gru_cell_1/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:1*
T0*
_output_shapes
:
b
AssignVariableOp_1AssignVariableOp!while/gru_cell_1/recurrent_kernel
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:2*
T0*
_output_shapes
:
V
AssignVariableOp_2AssignVariableOpwhile/gru_cell_1/bias
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:3*
T0*
_output_shapes
:
U
AssignVariableOp_3AssignVariableOpwhile/dense_2/kernel
Identity_4*
dtype0
F

Identity_5IdentityRestoreV2:4*
T0*
_output_shapes
:
S
AssignVariableOp_4AssignVariableOpwhile/dense_2/bias
Identity_5*
dtype0
F

Identity_6IdentityRestoreV2:5*
T0*
_output_shapes
:
U
AssignVariableOp_5AssignVariableOpwhile/dense_3/kernel
Identity_6*
dtype0
F

Identity_7IdentityRestoreV2:6*
T0*
_output_shapes
:
S
AssignVariableOp_6AssignVariableOpwhile/dense_3/bias
Identity_7*
dtype0
F

Identity_8IdentityRestoreV2:7*
T0*
_output_shapes
:
U
AssignVariableOp_7AssignVariableOpwhile/dense_4/kernel
Identity_8*
dtype0
F

Identity_9IdentityRestoreV2:8*
T0*
_output_shapes
:
S
AssignVariableOp_8AssignVariableOpwhile/dense_4/bias
Identity_9*
dtype0
G
Identity_10IdentityRestoreV2:9*
T0*
_output_shapes
:
V
AssignVariableOp_9AssignVariableOpwhile/dense_5/kernelIdentity_10*
dtype0
H
Identity_11IdentityRestoreV2:10*
T0*
_output_shapes
:
U
AssignVariableOp_10AssignVariableOpwhile/dense_5/biasIdentity_11*
dtype0
H
Identity_12IdentityRestoreV2:11*
T0*
_output_shapes
:
M
AssignVariableOp_11AssignVariableOp
rnn/kernelIdentity_12*
dtype0
H
Identity_13IdentityRestoreV2:12*
T0*
_output_shapes
:
W
AssignVariableOp_12AssignVariableOprnn/recurrent_kernelIdentity_13*
dtype0
H
Identity_14IdentityRestoreV2:13*
T0*
_output_shapes
:
K
AssignVariableOp_13AssignVariableOprnn/biasIdentity_14*
dtype0

RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 

RestoreV2_1	RestoreV2saver_filenameRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2

NoOp_1NoOp"/device:CPU:0

Identity_15Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: Хї
ѓ
ш
while_cond_200381
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_200381___redundant_placeholder0.
*while_cond_200381___redundant_placeholder1.
*while_cond_200381___redundant_placeholder2.
*while_cond_200381___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::


rnn_while_cond_199544
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_rnn_strided_slice_12
.rnn_while_cond_199544___redundant_placeholder02
.rnn_while_cond_199544___redundant_placeholder12
.rnn_while_cond_199544___redundant_placeholder22
.rnn_while_cond_199544___redundant_placeholder3
identity
\
LessLessplaceholderless_rnn_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
ФR
Ђ
?__inference_rnn_layer_call_and_return_conditional_losses_200949
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_200859*
condR
while_cond_200858*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1Л
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
б

+__inference_gru_cell_1_layer_call_fn_201586

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
і1
д
rnn_while_body_199226
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
rnn_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
rnn_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЙ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/yb
add_5AddV2rnn_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityІ

Identity_1Identityrnn_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0",
rnn_strided_slice_1rnn_strided_slice_1_0"Є
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
ш
м
C__inference_dense_2_layer_call_and_return_conditional_losses_201596

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Й
p
R__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_200148
inputs_0
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
Э
С
(__inference_dense_5_layer_call_fn_201666

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ъ
Ћ
D__inference_gru_cell_layer_call_and_return_conditional_losses_201774

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
ш
м
C__inference_dense_5_layer_call_and_return_conditional_losses_201656

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
НШ
б
!__inference__wrapped_model_197082
input_1'
#seq2seq_rnn_readvariableop_resource.
*seq2seq_rnn_matmul_readvariableop_resource0
,seq2seq_rnn_matmul_1_readvariableop_resource.
*seq2seq_gru_cell_1_readvariableop_resource5
1seq2seq_gru_cell_1_matmul_readvariableop_resource7
3seq2seq_gru_cell_1_matmul_1_readvariableop_resource2
.seq2seq_dense_2_matmul_readvariableop_resource3
/seq2seq_dense_2_biasadd_readvariableop_resource2
.seq2seq_dense_3_matmul_readvariableop_resource3
/seq2seq_dense_3_biasadd_readvariableop_resource2
.seq2seq_dense_4_matmul_readvariableop_resource3
/seq2seq_dense_4_biasadd_readvariableop_resource2
.seq2seq_dense_5_matmul_readvariableop_resource3
/seq2seq_dense_5_biasadd_readvariableop_resource
identityЂ&seq2seq/dense_2/BiasAdd/ReadVariableOpЂ%seq2seq/dense_2/MatMul/ReadVariableOpЂ&seq2seq/dense_3/BiasAdd/ReadVariableOpЂ%seq2seq/dense_3/MatMul/ReadVariableOpЂ&seq2seq/dense_4/BiasAdd/ReadVariableOpЂ%seq2seq/dense_4/MatMul/ReadVariableOpЂ&seq2seq/dense_5/BiasAdd/ReadVariableOpЂ%seq2seq/dense_5/MatMul/ReadVariableOpЂ(seq2seq/gru_cell_1/MatMul/ReadVariableOpЂ*seq2seq/gru_cell_1/MatMul_1/ReadVariableOpЂ!seq2seq/gru_cell_1/ReadVariableOpЂ*seq2seq/gru_cell_1_1/MatMul/ReadVariableOpЂ,seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOpЂ#seq2seq/gru_cell_1_1/ReadVariableOpЂ*seq2seq/gru_cell_1_2/MatMul/ReadVariableOpЂ,seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOpЂ#seq2seq/gru_cell_1_2/ReadVariableOpЂ*seq2seq/gru_cell_1_3/MatMul/ReadVariableOpЂ,seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOpЂ#seq2seq/gru_cell_1_3/ReadVariableOpЂ!seq2seq/rnn/MatMul/ReadVariableOpЂ#seq2seq/rnn/MatMul_1/ReadVariableOpЂseq2seq/rnn/ReadVariableOpЂseq2seq/rnn/whileУ
5seq2seq/tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    27
5seq2seq/tf_op_layer_strided_slice/strided_slice/beginП
3seq2seq/tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           25
3seq2seq/tf_op_layer_strided_slice/strided_slice/endЧ
7seq2seq/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         29
7seq2seq/tf_op_layer_strided_slice/strided_slice/stridesП
/seq2seq/tf_op_layer_strided_slice/strided_sliceStridedSliceinput_1>seq2seq/tf_op_layer_strided_slice/strided_slice/begin:output:0<seq2seq/tf_op_layer_strided_slice/strided_slice/end:output:0@seq2seq/tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask21
/seq2seq/tf_op_layer_strided_slice/strided_slice 
-seq2seq/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-seq2seq/tf_op_layer_ExpandDims/ExpandDims/dim
)seq2seq/tf_op_layer_ExpandDims/ExpandDims
ExpandDims8seq2seq/tf_op_layer_strided_slice/strided_slice:output:06seq2seq/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2+
)seq2seq/tf_op_layer_ExpandDims/ExpandDims]
seq2seq/rnn/ShapeShapeinput_1*
T0*
_output_shapes
:2
seq2seq/rnn/Shape
seq2seq/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
seq2seq/rnn/strided_slice/stack
!seq2seq/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!seq2seq/rnn/strided_slice/stack_1
!seq2seq/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!seq2seq/rnn/strided_slice/stack_2Њ
seq2seq/rnn/strided_sliceStridedSliceseq2seq/rnn/Shape:output:0(seq2seq/rnn/strided_slice/stack:output:0*seq2seq/rnn/strided_slice/stack_1:output:0*seq2seq/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
seq2seq/rnn/strided_slicet
seq2seq/rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
seq2seq/rnn/zeros/mul/y
seq2seq/rnn/zeros/mulMul"seq2seq/rnn/strided_slice:output:0 seq2seq/rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
seq2seq/rnn/zeros/mulw
seq2seq/rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
seq2seq/rnn/zeros/Less/y
seq2seq/rnn/zeros/LessLessseq2seq/rnn/zeros/mul:z:0!seq2seq/rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
seq2seq/rnn/zeros/Lessz
seq2seq/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
seq2seq/rnn/zeros/packed/1Г
seq2seq/rnn/zeros/packedPack"seq2seq/rnn/strided_slice:output:0#seq2seq/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
seq2seq/rnn/zeros/packedw
seq2seq/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
seq2seq/rnn/zeros/ConstЅ
seq2seq/rnn/zerosFill!seq2seq/rnn/zeros/packed:output:0 seq2seq/rnn/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/zeros
seq2seq/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
seq2seq/rnn/transpose/perm
seq2seq/rnn/transpose	Transposeinput_1#seq2seq/rnn/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
seq2seq/rnn/transposes
seq2seq/rnn/Shape_1Shapeseq2seq/rnn/transpose:y:0*
T0*
_output_shapes
:2
seq2seq/rnn/Shape_1
!seq2seq/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!seq2seq/rnn/strided_slice_1/stack
#seq2seq/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#seq2seq/rnn/strided_slice_1/stack_1
#seq2seq/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#seq2seq/rnn/strided_slice_1/stack_2Ж
seq2seq/rnn/strided_slice_1StridedSliceseq2seq/rnn/Shape_1:output:0*seq2seq/rnn/strided_slice_1/stack:output:0,seq2seq/rnn/strided_slice_1/stack_1:output:0,seq2seq/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
seq2seq/rnn/strided_slice_1
'seq2seq/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'seq2seq/rnn/TensorArrayV2/element_shapeт
seq2seq/rnn/TensorArrayV2TensorListReserve0seq2seq/rnn/TensorArrayV2/element_shape:output:0$seq2seq/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
seq2seq/rnn/TensorArrayV2з
Aseq2seq/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2C
Aseq2seq/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeЈ
3seq2seq/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorseq2seq/rnn/transpose:y:0Jseq2seq/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type025
3seq2seq/rnn/TensorArrayUnstack/TensorListFromTensor
!seq2seq/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!seq2seq/rnn/strided_slice_2/stack
#seq2seq/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#seq2seq/rnn/strided_slice_2/stack_1
#seq2seq/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#seq2seq/rnn/strided_slice_2/stack_2Ф
seq2seq/rnn/strided_slice_2StridedSliceseq2seq/rnn/transpose:y:0*seq2seq/rnn/strided_slice_2/stack:output:0,seq2seq/rnn/strided_slice_2/stack_1:output:0,seq2seq/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
seq2seq/rnn/strided_slice_2
seq2seq/rnn/ReadVariableOpReadVariableOp#seq2seq_rnn_readvariableop_resource*
_output_shapes

:`*
dtype02
seq2seq/rnn/ReadVariableOp
seq2seq/rnn/unstackUnpack"seq2seq/rnn/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
seq2seq/rnn/unstackБ
!seq2seq/rnn/MatMul/ReadVariableOpReadVariableOp*seq2seq_rnn_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02#
!seq2seq/rnn/MatMul/ReadVariableOpЕ
seq2seq/rnn/MatMulMatMul$seq2seq/rnn/strided_slice_2:output:0)seq2seq/rnn/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/rnn/MatMulЃ
seq2seq/rnn/BiasAddBiasAddseq2seq/rnn/MatMul:product:0seq2seq/rnn/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/rnn/BiasAddh
seq2seq/rnn/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
seq2seq/rnn/Const
seq2seq/rnn/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
seq2seq/rnn/split/split_dimм
seq2seq/rnn/splitSplit$seq2seq/rnn/split/split_dim:output:0seq2seq/rnn/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/rnn/splitЗ
#seq2seq/rnn/MatMul_1/ReadVariableOpReadVariableOp,seq2seq_rnn_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02%
#seq2seq/rnn/MatMul_1/ReadVariableOpБ
seq2seq/rnn/MatMul_1MatMulseq2seq/rnn/zeros:output:0+seq2seq/rnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/rnn/MatMul_1Љ
seq2seq/rnn/BiasAdd_1BiasAddseq2seq/rnn/MatMul_1:product:0seq2seq/rnn/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/rnn/BiasAdd_1
seq2seq/rnn/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
seq2seq/rnn/Const_1
seq2seq/rnn/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
seq2seq/rnn/split_1/split_dim
seq2seq/rnn/split_1SplitVseq2seq/rnn/BiasAdd_1:output:0seq2seq/rnn/Const_1:output:0&seq2seq/rnn/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/rnn/split_1
seq2seq/rnn/addAddV2seq2seq/rnn/split:output:0seq2seq/rnn/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/add|
seq2seq/rnn/SigmoidSigmoidseq2seq/rnn/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/Sigmoid
seq2seq/rnn/add_1AddV2seq2seq/rnn/split:output:1seq2seq/rnn/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/add_1
seq2seq/rnn/Sigmoid_1Sigmoidseq2seq/rnn/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/Sigmoid_1
seq2seq/rnn/mulMulseq2seq/rnn/Sigmoid_1:y:0seq2seq/rnn/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/mul
seq2seq/rnn/add_2AddV2seq2seq/rnn/split:output:2seq2seq/rnn/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/add_2u
seq2seq/rnn/TanhTanhseq2seq/rnn/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/Tanh
seq2seq/rnn/mul_1Mulseq2seq/rnn/Sigmoid:y:0seq2seq/rnn/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/mul_1k
seq2seq/rnn/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
seq2seq/rnn/sub/x
seq2seq/rnn/subSubseq2seq/rnn/sub/x:output:0seq2seq/rnn/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/sub
seq2seq/rnn/mul_2Mulseq2seq/rnn/sub:z:0seq2seq/rnn/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/mul_2
seq2seq/rnn/add_3AddV2seq2seq/rnn/mul_1:z:0seq2seq/rnn/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/add_3Ї
)seq2seq/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2+
)seq2seq/rnn/TensorArrayV2_1/element_shapeш
seq2seq/rnn/TensorArrayV2_1TensorListReserve2seq2seq/rnn/TensorArrayV2_1/element_shape:output:0$seq2seq/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
seq2seq/rnn/TensorArrayV2_1f
seq2seq/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
seq2seq/rnn/time
$seq2seq/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$seq2seq/rnn/while/maximum_iterations
seq2seq/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2 
seq2seq/rnn/while/loop_counterю
seq2seq/rnn/whileWhile'seq2seq/rnn/while/loop_counter:output:0-seq2seq/rnn/while/maximum_iterations:output:0seq2seq/rnn/time:output:0$seq2seq/rnn/TensorArrayV2_1:handle:0seq2seq/rnn/zeros:output:0$seq2seq/rnn/strided_slice_1:output:0Cseq2seq/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0#seq2seq_rnn_readvariableop_resource*seq2seq_rnn_matmul_readvariableop_resource,seq2seq_rnn_matmul_1_readvariableop_resource"^seq2seq/rnn/MatMul/ReadVariableOp$^seq2seq/rnn/MatMul_1/ReadVariableOp^seq2seq/rnn/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *)
body!R
seq2seq_rnn_while_body_196839*)
cond!R
seq2seq_rnn_while_cond_196838*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
seq2seq/rnn/whileЭ
<seq2seq/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2>
<seq2seq/rnn/TensorArrayV2Stack/TensorListStack/element_shape
.seq2seq/rnn/TensorArrayV2Stack/TensorListStackTensorListStackseq2seq/rnn/while:output:3Eseq2seq/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype020
.seq2seq/rnn/TensorArrayV2Stack/TensorListStack
!seq2seq/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2#
!seq2seq/rnn/strided_slice_3/stack
#seq2seq/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#seq2seq/rnn/strided_slice_3/stack_1
#seq2seq/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#seq2seq/rnn/strided_slice_3/stack_2т
seq2seq/rnn/strided_slice_3StridedSlice7seq2seq/rnn/TensorArrayV2Stack/TensorListStack:tensor:0*seq2seq/rnn/strided_slice_3/stack:output:0,seq2seq/rnn/strided_slice_3/stack_1:output:0,seq2seq/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
seq2seq/rnn/strided_slice_3
seq2seq/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
seq2seq/rnn/transpose_1/permе
seq2seq/rnn/transpose_1	Transpose7seq2seq/rnn/TensorArrayV2Stack/TensorListStack:tensor:0%seq2seq/rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
seq2seq/rnn/transpose_1Б
!seq2seq/gru_cell_1/ReadVariableOpReadVariableOp*seq2seq_gru_cell_1_readvariableop_resource*
_output_shapes

:`*
dtype02#
!seq2seq/gru_cell_1/ReadVariableOpЃ
seq2seq/gru_cell_1/unstackUnpack)seq2seq/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
seq2seq/gru_cell_1/unstackЦ
(seq2seq/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1seq2seq_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02*
(seq2seq/gru_cell_1/MatMul/ReadVariableOpи
seq2seq/gru_cell_1/MatMulMatMul2seq2seq/tf_op_layer_ExpandDims/ExpandDims:output:00seq2seq/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1/MatMulП
seq2seq/gru_cell_1/BiasAddBiasAdd#seq2seq/gru_cell_1/MatMul:product:0#seq2seq/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1/BiasAddv
seq2seq/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
seq2seq/gru_cell_1/Const
"seq2seq/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2$
"seq2seq/gru_cell_1/split/split_dimј
seq2seq/gru_cell_1/splitSplit+seq2seq/gru_cell_1/split/split_dim:output:0#seq2seq/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1/splitЬ
*seq2seq/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3seq2seq_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02,
*seq2seq/gru_cell_1/MatMul_1/ReadVariableOpЦ
seq2seq/gru_cell_1/MatMul_1MatMulseq2seq/rnn/while:output:42seq2seq/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1/MatMul_1Х
seq2seq/gru_cell_1/BiasAdd_1BiasAdd%seq2seq/gru_cell_1/MatMul_1:product:0#seq2seq/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1/BiasAdd_1
seq2seq/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
seq2seq/gru_cell_1/Const_1
$seq2seq/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$seq2seq/gru_cell_1/split_1/split_dimВ
seq2seq/gru_cell_1/split_1SplitV%seq2seq/gru_cell_1/BiasAdd_1:output:0#seq2seq/gru_cell_1/Const_1:output:0-seq2seq/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1/split_1Г
seq2seq/gru_cell_1/addAddV2!seq2seq/gru_cell_1/split:output:0#seq2seq/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/add
seq2seq/gru_cell_1/SigmoidSigmoidseq2seq/gru_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/SigmoidЗ
seq2seq/gru_cell_1/add_1AddV2!seq2seq/gru_cell_1/split:output:1#seq2seq/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/add_1
seq2seq/gru_cell_1/Sigmoid_1Sigmoidseq2seq/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/Sigmoid_1А
seq2seq/gru_cell_1/mulMul seq2seq/gru_cell_1/Sigmoid_1:y:0#seq2seq/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/mulЎ
seq2seq/gru_cell_1/add_2AddV2!seq2seq/gru_cell_1/split:output:2seq2seq/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/add_2
seq2seq/gru_cell_1/TanhTanhseq2seq/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/TanhЉ
seq2seq/gru_cell_1/mul_1Mulseq2seq/gru_cell_1/Sigmoid:y:0seq2seq/rnn/while:output:4*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/mul_1y
seq2seq/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
seq2seq/gru_cell_1/sub/xЌ
seq2seq/gru_cell_1/subSub!seq2seq/gru_cell_1/sub/x:output:0seq2seq/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/subІ
seq2seq/gru_cell_1/mul_2Mulseq2seq/gru_cell_1/sub:z:0seq2seq/gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/mul_2Ћ
seq2seq/gru_cell_1/add_3AddV2seq2seq/gru_cell_1/mul_1:z:0seq2seq/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1/add_3Н
%seq2seq/dense_2/MatMul/ReadVariableOpReadVariableOp.seq2seq_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%seq2seq/dense_2/MatMul/ReadVariableOpЙ
seq2seq/dense_2/MatMulMatMulseq2seq/gru_cell_1/add_3:z:0-seq2seq/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_2/MatMulМ
&seq2seq/dense_2/BiasAdd/ReadVariableOpReadVariableOp/seq2seq_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&seq2seq/dense_2/BiasAdd/ReadVariableOpС
seq2seq/dense_2/BiasAddBiasAdd seq2seq/dense_2/MatMul:product:0.seq2seq/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_2/BiasAddй
#seq2seq/gru_cell_1_1/ReadVariableOpReadVariableOp*seq2seq_gru_cell_1_readvariableop_resource"^seq2seq/gru_cell_1/ReadVariableOp*
_output_shapes

:`*
dtype02%
#seq2seq/gru_cell_1_1/ReadVariableOpЉ
seq2seq/gru_cell_1_1/unstackUnpack+seq2seq/gru_cell_1_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
seq2seq/gru_cell_1_1/unstackѕ
*seq2seq/gru_cell_1_1/MatMul/ReadVariableOpReadVariableOp1seq2seq_gru_cell_1_matmul_readvariableop_resource)^seq2seq/gru_cell_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02,
*seq2seq/gru_cell_1_1/MatMul/ReadVariableOpЬ
seq2seq/gru_cell_1_1/MatMulMatMul seq2seq/dense_2/BiasAdd:output:02seq2seq/gru_cell_1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_1/MatMulЧ
seq2seq/gru_cell_1_1/BiasAddBiasAdd%seq2seq/gru_cell_1_1/MatMul:product:0%seq2seq/gru_cell_1_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_1/BiasAddz
seq2seq/gru_cell_1_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
seq2seq/gru_cell_1_1/Const
$seq2seq/gru_cell_1_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$seq2seq/gru_cell_1_1/split/split_dim
seq2seq/gru_cell_1_1/splitSplit-seq2seq/gru_cell_1_1/split/split_dim:output:0%seq2seq/gru_cell_1_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_1/split§
,seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOpReadVariableOp3seq2seq_gru_cell_1_matmul_1_readvariableop_resource+^seq2seq/gru_cell_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02.
,seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOpЮ
seq2seq/gru_cell_1_1/MatMul_1MatMulseq2seq/gru_cell_1/add_3:z:04seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_1/MatMul_1Э
seq2seq/gru_cell_1_1/BiasAdd_1BiasAdd'seq2seq/gru_cell_1_1/MatMul_1:product:0%seq2seq/gru_cell_1_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2 
seq2seq/gru_cell_1_1/BiasAdd_1
seq2seq/gru_cell_1_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
seq2seq/gru_cell_1_1/Const_1
&seq2seq/gru_cell_1_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&seq2seq/gru_cell_1_1/split_1/split_dimМ
seq2seq/gru_cell_1_1/split_1SplitV'seq2seq/gru_cell_1_1/BiasAdd_1:output:0%seq2seq/gru_cell_1_1/Const_1:output:0/seq2seq/gru_cell_1_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_1/split_1Л
seq2seq/gru_cell_1_1/addAddV2#seq2seq/gru_cell_1_1/split:output:0%seq2seq/gru_cell_1_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/add
seq2seq/gru_cell_1_1/SigmoidSigmoidseq2seq/gru_cell_1_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/SigmoidП
seq2seq/gru_cell_1_1/add_1AddV2#seq2seq/gru_cell_1_1/split:output:1%seq2seq/gru_cell_1_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/add_1
seq2seq/gru_cell_1_1/Sigmoid_1Sigmoidseq2seq/gru_cell_1_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
seq2seq/gru_cell_1_1/Sigmoid_1И
seq2seq/gru_cell_1_1/mulMul"seq2seq/gru_cell_1_1/Sigmoid_1:y:0%seq2seq/gru_cell_1_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/mulЖ
seq2seq/gru_cell_1_1/add_2AddV2#seq2seq/gru_cell_1_1/split:output:2seq2seq/gru_cell_1_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/add_2
seq2seq/gru_cell_1_1/TanhTanhseq2seq/gru_cell_1_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/TanhБ
seq2seq/gru_cell_1_1/mul_1Mul seq2seq/gru_cell_1_1/Sigmoid:y:0seq2seq/gru_cell_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/mul_1}
seq2seq/gru_cell_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
seq2seq/gru_cell_1_1/sub/xД
seq2seq/gru_cell_1_1/subSub#seq2seq/gru_cell_1_1/sub/x:output:0 seq2seq/gru_cell_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/subЎ
seq2seq/gru_cell_1_1/mul_2Mulseq2seq/gru_cell_1_1/sub:z:0seq2seq/gru_cell_1_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/mul_2Г
seq2seq/gru_cell_1_1/add_3AddV2seq2seq/gru_cell_1_1/mul_1:z:0seq2seq/gru_cell_1_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_1/add_3Н
%seq2seq/dense_3/MatMul/ReadVariableOpReadVariableOp.seq2seq_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%seq2seq/dense_3/MatMul/ReadVariableOpЛ
seq2seq/dense_3/MatMulMatMulseq2seq/gru_cell_1_1/add_3:z:0-seq2seq/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_3/MatMulМ
&seq2seq/dense_3/BiasAdd/ReadVariableOpReadVariableOp/seq2seq_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&seq2seq/dense_3/BiasAdd/ReadVariableOpС
seq2seq/dense_3/BiasAddBiasAdd seq2seq/dense_3/MatMul:product:0.seq2seq/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_3/BiasAddл
#seq2seq/gru_cell_1_2/ReadVariableOpReadVariableOp*seq2seq_gru_cell_1_readvariableop_resource$^seq2seq/gru_cell_1_1/ReadVariableOp*
_output_shapes

:`*
dtype02%
#seq2seq/gru_cell_1_2/ReadVariableOpЉ
seq2seq/gru_cell_1_2/unstackUnpack+seq2seq/gru_cell_1_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
seq2seq/gru_cell_1_2/unstackї
*seq2seq/gru_cell_1_2/MatMul/ReadVariableOpReadVariableOp1seq2seq_gru_cell_1_matmul_readvariableop_resource+^seq2seq/gru_cell_1_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02,
*seq2seq/gru_cell_1_2/MatMul/ReadVariableOpЬ
seq2seq/gru_cell_1_2/MatMulMatMul seq2seq/dense_3/BiasAdd:output:02seq2seq/gru_cell_1_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_2/MatMulЧ
seq2seq/gru_cell_1_2/BiasAddBiasAdd%seq2seq/gru_cell_1_2/MatMul:product:0%seq2seq/gru_cell_1_2/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_2/BiasAddz
seq2seq/gru_cell_1_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
seq2seq/gru_cell_1_2/Const
$seq2seq/gru_cell_1_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$seq2seq/gru_cell_1_2/split/split_dim
seq2seq/gru_cell_1_2/splitSplit-seq2seq/gru_cell_1_2/split/split_dim:output:0%seq2seq/gru_cell_1_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_2/splitџ
,seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOpReadVariableOp3seq2seq_gru_cell_1_matmul_1_readvariableop_resource-^seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02.
,seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOpа
seq2seq/gru_cell_1_2/MatMul_1MatMulseq2seq/gru_cell_1_1/add_3:z:04seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_2/MatMul_1Э
seq2seq/gru_cell_1_2/BiasAdd_1BiasAdd'seq2seq/gru_cell_1_2/MatMul_1:product:0%seq2seq/gru_cell_1_2/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2 
seq2seq/gru_cell_1_2/BiasAdd_1
seq2seq/gru_cell_1_2/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
seq2seq/gru_cell_1_2/Const_1
&seq2seq/gru_cell_1_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&seq2seq/gru_cell_1_2/split_1/split_dimМ
seq2seq/gru_cell_1_2/split_1SplitV'seq2seq/gru_cell_1_2/BiasAdd_1:output:0%seq2seq/gru_cell_1_2/Const_1:output:0/seq2seq/gru_cell_1_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_2/split_1Л
seq2seq/gru_cell_1_2/addAddV2#seq2seq/gru_cell_1_2/split:output:0%seq2seq/gru_cell_1_2/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/add
seq2seq/gru_cell_1_2/SigmoidSigmoidseq2seq/gru_cell_1_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/SigmoidП
seq2seq/gru_cell_1_2/add_1AddV2#seq2seq/gru_cell_1_2/split:output:1%seq2seq/gru_cell_1_2/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/add_1
seq2seq/gru_cell_1_2/Sigmoid_1Sigmoidseq2seq/gru_cell_1_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
seq2seq/gru_cell_1_2/Sigmoid_1И
seq2seq/gru_cell_1_2/mulMul"seq2seq/gru_cell_1_2/Sigmoid_1:y:0%seq2seq/gru_cell_1_2/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/mulЖ
seq2seq/gru_cell_1_2/add_2AddV2#seq2seq/gru_cell_1_2/split:output:2seq2seq/gru_cell_1_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/add_2
seq2seq/gru_cell_1_2/TanhTanhseq2seq/gru_cell_1_2/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/TanhГ
seq2seq/gru_cell_1_2/mul_1Mul seq2seq/gru_cell_1_2/Sigmoid:y:0seq2seq/gru_cell_1_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/mul_1}
seq2seq/gru_cell_1_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
seq2seq/gru_cell_1_2/sub/xД
seq2seq/gru_cell_1_2/subSub#seq2seq/gru_cell_1_2/sub/x:output:0 seq2seq/gru_cell_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/subЎ
seq2seq/gru_cell_1_2/mul_2Mulseq2seq/gru_cell_1_2/sub:z:0seq2seq/gru_cell_1_2/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/mul_2Г
seq2seq/gru_cell_1_2/add_3AddV2seq2seq/gru_cell_1_2/mul_1:z:0seq2seq/gru_cell_1_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_2/add_3Н
%seq2seq/dense_4/MatMul/ReadVariableOpReadVariableOp.seq2seq_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%seq2seq/dense_4/MatMul/ReadVariableOpЛ
seq2seq/dense_4/MatMulMatMulseq2seq/gru_cell_1_2/add_3:z:0-seq2seq/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_4/MatMulМ
&seq2seq/dense_4/BiasAdd/ReadVariableOpReadVariableOp/seq2seq_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&seq2seq/dense_4/BiasAdd/ReadVariableOpС
seq2seq/dense_4/BiasAddBiasAdd seq2seq/dense_4/MatMul:product:0.seq2seq/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_4/BiasAddл
#seq2seq/gru_cell_1_3/ReadVariableOpReadVariableOp*seq2seq_gru_cell_1_readvariableop_resource$^seq2seq/gru_cell_1_2/ReadVariableOp*
_output_shapes

:`*
dtype02%
#seq2seq/gru_cell_1_3/ReadVariableOpЉ
seq2seq/gru_cell_1_3/unstackUnpack+seq2seq/gru_cell_1_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
seq2seq/gru_cell_1_3/unstackї
*seq2seq/gru_cell_1_3/MatMul/ReadVariableOpReadVariableOp1seq2seq_gru_cell_1_matmul_readvariableop_resource+^seq2seq/gru_cell_1_2/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02,
*seq2seq/gru_cell_1_3/MatMul/ReadVariableOpЬ
seq2seq/gru_cell_1_3/MatMulMatMul seq2seq/dense_4/BiasAdd:output:02seq2seq/gru_cell_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_3/MatMulЧ
seq2seq/gru_cell_1_3/BiasAddBiasAdd%seq2seq/gru_cell_1_3/MatMul:product:0%seq2seq/gru_cell_1_3/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_3/BiasAddz
seq2seq/gru_cell_1_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
seq2seq/gru_cell_1_3/Const
$seq2seq/gru_cell_1_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2&
$seq2seq/gru_cell_1_3/split/split_dim
seq2seq/gru_cell_1_3/splitSplit-seq2seq/gru_cell_1_3/split/split_dim:output:0%seq2seq/gru_cell_1_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_3/splitџ
,seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOpReadVariableOp3seq2seq_gru_cell_1_matmul_1_readvariableop_resource-^seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02.
,seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOpа
seq2seq/gru_cell_1_3/MatMul_1MatMulseq2seq/gru_cell_1_2/add_3:z:04seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
seq2seq/gru_cell_1_3/MatMul_1Э
seq2seq/gru_cell_1_3/BiasAdd_1BiasAdd'seq2seq/gru_cell_1_3/MatMul_1:product:0%seq2seq/gru_cell_1_3/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2 
seq2seq/gru_cell_1_3/BiasAdd_1
seq2seq/gru_cell_1_3/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
seq2seq/gru_cell_1_3/Const_1
&seq2seq/gru_cell_1_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2(
&seq2seq/gru_cell_1_3/split_1/split_dimМ
seq2seq/gru_cell_1_3/split_1SplitV'seq2seq/gru_cell_1_3/BiasAdd_1:output:0%seq2seq/gru_cell_1_3/Const_1:output:0/seq2seq/gru_cell_1_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
seq2seq/gru_cell_1_3/split_1Л
seq2seq/gru_cell_1_3/addAddV2#seq2seq/gru_cell_1_3/split:output:0%seq2seq/gru_cell_1_3/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/add
seq2seq/gru_cell_1_3/SigmoidSigmoidseq2seq/gru_cell_1_3/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/SigmoidП
seq2seq/gru_cell_1_3/add_1AddV2#seq2seq/gru_cell_1_3/split:output:1%seq2seq/gru_cell_1_3/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/add_1
seq2seq/gru_cell_1_3/Sigmoid_1Sigmoidseq2seq/gru_cell_1_3/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2 
seq2seq/gru_cell_1_3/Sigmoid_1И
seq2seq/gru_cell_1_3/mulMul"seq2seq/gru_cell_1_3/Sigmoid_1:y:0%seq2seq/gru_cell_1_3/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/mulЖ
seq2seq/gru_cell_1_3/add_2AddV2#seq2seq/gru_cell_1_3/split:output:2seq2seq/gru_cell_1_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/add_2
seq2seq/gru_cell_1_3/TanhTanhseq2seq/gru_cell_1_3/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/TanhГ
seq2seq/gru_cell_1_3/mul_1Mul seq2seq/gru_cell_1_3/Sigmoid:y:0seq2seq/gru_cell_1_2/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/mul_1}
seq2seq/gru_cell_1_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
seq2seq/gru_cell_1_3/sub/xД
seq2seq/gru_cell_1_3/subSub#seq2seq/gru_cell_1_3/sub/x:output:0 seq2seq/gru_cell_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/subЎ
seq2seq/gru_cell_1_3/mul_2Mulseq2seq/gru_cell_1_3/sub:z:0seq2seq/gru_cell_1_3/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/mul_2Г
seq2seq/gru_cell_1_3/add_3AddV2seq2seq/gru_cell_1_3/mul_1:z:0seq2seq/gru_cell_1_3/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
seq2seq/gru_cell_1_3/add_3Н
%seq2seq/dense_5/MatMul/ReadVariableOpReadVariableOp.seq2seq_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%seq2seq/dense_5/MatMul/ReadVariableOpЛ
seq2seq/dense_5/MatMulMatMulseq2seq/gru_cell_1_3/add_3:z:0-seq2seq/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_5/MatMulМ
&seq2seq/dense_5/BiasAdd/ReadVariableOpReadVariableOp/seq2seq_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&seq2seq/dense_5/BiasAdd/ReadVariableOpС
seq2seq/dense_5/BiasAddBiasAdd seq2seq/dense_5/MatMul:product:0.seq2seq/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
seq2seq/dense_5/BiasAddЄ
!seq2seq/tf_op_layer_packed/packedPack seq2seq/dense_2/BiasAdd:output:0 seq2seq/dense_3/BiasAdd:output:0 seq2seq/dense_4/BiasAdd:output:0 seq2seq/dense_5/BiasAdd:output:0*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2#
!seq2seq/tf_op_layer_packed/packedБ
,seq2seq/tf_op_layer_transpose/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,seq2seq/tf_op_layer_transpose/transpose/perm
'seq2seq/tf_op_layer_transpose/transpose	Transpose*seq2seq/tf_op_layer_packed/packed:output:05seq2seq/tf_op_layer_transpose/transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2)
'seq2seq/tf_op_layer_transpose/transposeФ
IdentityIdentity+seq2seq/tf_op_layer_transpose/transpose:y:0'^seq2seq/dense_2/BiasAdd/ReadVariableOp&^seq2seq/dense_2/MatMul/ReadVariableOp'^seq2seq/dense_3/BiasAdd/ReadVariableOp&^seq2seq/dense_3/MatMul/ReadVariableOp'^seq2seq/dense_4/BiasAdd/ReadVariableOp&^seq2seq/dense_4/MatMul/ReadVariableOp'^seq2seq/dense_5/BiasAdd/ReadVariableOp&^seq2seq/dense_5/MatMul/ReadVariableOp)^seq2seq/gru_cell_1/MatMul/ReadVariableOp+^seq2seq/gru_cell_1/MatMul_1/ReadVariableOp"^seq2seq/gru_cell_1/ReadVariableOp+^seq2seq/gru_cell_1_1/MatMul/ReadVariableOp-^seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOp$^seq2seq/gru_cell_1_1/ReadVariableOp+^seq2seq/gru_cell_1_2/MatMul/ReadVariableOp-^seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOp$^seq2seq/gru_cell_1_2/ReadVariableOp+^seq2seq/gru_cell_1_3/MatMul/ReadVariableOp-^seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOp$^seq2seq/gru_cell_1_3/ReadVariableOp"^seq2seq/rnn/MatMul/ReadVariableOp$^seq2seq/rnn/MatMul_1/ReadVariableOp^seq2seq/rnn/ReadVariableOp^seq2seq/rnn/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::2P
&seq2seq/dense_2/BiasAdd/ReadVariableOp&seq2seq/dense_2/BiasAdd/ReadVariableOp2N
%seq2seq/dense_2/MatMul/ReadVariableOp%seq2seq/dense_2/MatMul/ReadVariableOp2P
&seq2seq/dense_3/BiasAdd/ReadVariableOp&seq2seq/dense_3/BiasAdd/ReadVariableOp2N
%seq2seq/dense_3/MatMul/ReadVariableOp%seq2seq/dense_3/MatMul/ReadVariableOp2P
&seq2seq/dense_4/BiasAdd/ReadVariableOp&seq2seq/dense_4/BiasAdd/ReadVariableOp2N
%seq2seq/dense_4/MatMul/ReadVariableOp%seq2seq/dense_4/MatMul/ReadVariableOp2P
&seq2seq/dense_5/BiasAdd/ReadVariableOp&seq2seq/dense_5/BiasAdd/ReadVariableOp2N
%seq2seq/dense_5/MatMul/ReadVariableOp%seq2seq/dense_5/MatMul/ReadVariableOp2T
(seq2seq/gru_cell_1/MatMul/ReadVariableOp(seq2seq/gru_cell_1/MatMul/ReadVariableOp2X
*seq2seq/gru_cell_1/MatMul_1/ReadVariableOp*seq2seq/gru_cell_1/MatMul_1/ReadVariableOp2F
!seq2seq/gru_cell_1/ReadVariableOp!seq2seq/gru_cell_1/ReadVariableOp2X
*seq2seq/gru_cell_1_1/MatMul/ReadVariableOp*seq2seq/gru_cell_1_1/MatMul/ReadVariableOp2\
,seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOp,seq2seq/gru_cell_1_1/MatMul_1/ReadVariableOp2J
#seq2seq/gru_cell_1_1/ReadVariableOp#seq2seq/gru_cell_1_1/ReadVariableOp2X
*seq2seq/gru_cell_1_2/MatMul/ReadVariableOp*seq2seq/gru_cell_1_2/MatMul/ReadVariableOp2\
,seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOp,seq2seq/gru_cell_1_2/MatMul_1/ReadVariableOp2J
#seq2seq/gru_cell_1_2/ReadVariableOp#seq2seq/gru_cell_1_2/ReadVariableOp2X
*seq2seq/gru_cell_1_3/MatMul/ReadVariableOp*seq2seq/gru_cell_1_3/MatMul/ReadVariableOp2\
,seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOp,seq2seq/gru_cell_1_3/MatMul_1/ReadVariableOp2J
#seq2seq/gru_cell_1_3/ReadVariableOp#seq2seq/gru_cell_1_3/ReadVariableOp2F
!seq2seq/rnn/MatMul/ReadVariableOp!seq2seq/rnn/MatMul/ReadVariableOp2J
#seq2seq/rnn/MatMul_1/ReadVariableOp#seq2seq/rnn/MatMul_1/ReadVariableOp28
seq2seq/rnn/ReadVariableOpseq2seq/rnn/ReadVariableOp2&
seq2seq/rnn/whileseq2seq/rnn/while:' #
!
_user_specified_name	input_1
і1
д
rnn_while_body_199545
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
rnn_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
rnn_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЙ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/yb
add_5AddV2rnn_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityІ

Identity_1Identityrnn_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0",
rnn_strided_slice_1rnn_strided_slice_1_0"Є
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
О1
И
while_body_200541
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
О1
И
while_body_200382
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
О1
И
while_body_200223
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
ЉR

$__inference_rnn_layer_call_fn_201267
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_201177*
condR
while_cond_201176*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1Л
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
б

+__inference_gru_cell_1_layer_call_fn_201546

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
в
o
Q__inference_tf_op_layer_transpose_layer_call_and_return_conditional_losses_201688
inputs_0
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
	transposee
IdentityIdentitytranspose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
Э
С
(__inference_dense_2_layer_call_fn_201606

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ѓQ

$__inference_rnn_layer_call_fn_200790

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_200700*
condR
while_cond_200699*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1В
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*+
_output_shapes
:џџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
ФR
Ђ
?__inference_rnn_layer_call_and_return_conditional_losses_201108
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_201018*
condR
while_cond_201017*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1Л
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0


rnn_while_cond_199225
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_rnn_strided_slice_12
.rnn_while_cond_199225___redundant_placeholder02
.rnn_while_cond_199225___redundant_placeholder12
.rnn_while_cond_199225___redundant_placeholder22
.rnn_while_cond_199225___redundant_placeholder3
identity
\
LessLessplaceholderless_rnn_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
Э
С
(__inference_dense_3_layer_call_fn_201626

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Я

)__inference_gru_cell_layer_call_fn_201814

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0

U
7__inference_tf_op_layer_ExpandDims_layer_call_fn_200154
inputs_0
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2

ExpandDimsg
IdentityIdentityExpandDims:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*"
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
ш
м
C__inference_dense_3_layer_call_and_return_conditional_losses_201616

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Э
С
(__inference_dense_4_layer_call_fn_201646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Л

N__inference_tf_op_layer_packed_layer_call_and_return_conditional_losses_201674
inputs_0
inputs_1
inputs_2
inputs_3
identity
packedPackinputs_0inputs_1inputs_2inputs_3*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
packedg
IdentityIdentitypacked:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3
О1
И
while_body_201018
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
О1
И
while_body_200700
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
Щ
Ј
(__inference_seq2seq_layer_call_fn_199788
input_1
rnn_readvariableop_resource&
"rnn_matmul_readvariableop_resource(
$rnn_matmul_1_readvariableop_resource&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂ gru_cell_1/MatMul/ReadVariableOpЂ"gru_cell_1/MatMul_1/ReadVariableOpЂgru_cell_1/ReadVariableOpЂ"gru_cell_1_1/MatMul/ReadVariableOpЂ$gru_cell_1_1/MatMul_1/ReadVariableOpЂgru_cell_1_1/ReadVariableOpЂ"gru_cell_1_2/MatMul/ReadVariableOpЂ$gru_cell_1_2/MatMul_1/ReadVariableOpЂgru_cell_1_2/ReadVariableOpЂ"gru_cell_1_3/MatMul/ReadVariableOpЂ$gru_cell_1_3/MatMul_1/ReadVariableOpЂgru_cell_1_3/ReadVariableOpЂrnn/MatMul/ReadVariableOpЂrnn/MatMul_1/ReadVariableOpЂrnn/ReadVariableOpЂ	rnn/whileГ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2/
-tf_op_layer_strided_slice/strided_slice/beginЏ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2-
+tf_op_layer_strided_slice/strided_slice/endЗ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/tf_op_layer_strided_slice/strided_slice/strides
'tf_op_layer_strided_slice/strided_sliceStridedSliceinput_16tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2)
'tf_op_layer_strided_slice/strided_slice
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimї
!tf_op_layer_ExpandDims/ExpandDims
ExpandDims0tf_op_layer_strided_slice/strided_slice:output:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2#
!tf_op_layer_ExpandDims/ExpandDimsM
	rnn/ShapeShapeinput_1*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2њ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/zeros}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposeinput_1rnn/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
rnn/TensorArrayV2/element_shapeТ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ч
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
rnn/strided_slice_2
rnn/ReadVariableOpReadVariableOprnn_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/ReadVariableOpv
rnn/unstackUnpackrnn/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
rnn/unstack
rnn/MatMul/ReadVariableOpReadVariableOp"rnn_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/MatMul/ReadVariableOp

rnn/MatMulMatMulrnn/strided_slice_2:output:0!rnn/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

rnn/MatMul
rnn/BiasAddBiasAddrnn/MatMul:product:0rnn/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAddX
	rnn/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
	rnn/Constu
rnn/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split/split_dimМ
	rnn/splitSplitrnn/split/split_dim:output:0rnn/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
	rnn/split
rnn/MatMul_1/ReadVariableOpReadVariableOp$rnn_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
rnn/MatMul_1/ReadVariableOp
rnn/MatMul_1MatMulrnn/zeros:output:0#rnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/MatMul_1
rnn/BiasAdd_1BiasAddrnn/MatMul_1:product:0rnn/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAdd_1o
rnn/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
rnn/Const_1y
rnn/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split_1/split_dimч
rnn/split_1SplitVrnn/BiasAdd_1:output:0rnn/Const_1:output:0rnn/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
rnn/split_1w
rnn/addAddV2rnn/split:output:0rnn/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/addd
rnn/SigmoidSigmoidrnn/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid{
	rnn/add_1AddV2rnn/split:output:1rnn/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_1j
rnn/Sigmoid_1Sigmoidrnn/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid_1t
rnn/mulMulrnn/Sigmoid_1:y:0rnn/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/mulr
	rnn/add_2AddV2rnn/split:output:2rnn/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_2]
rnn/TanhTanhrnn/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

rnn/Tanht
	rnn/mul_1Mulrnn/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_1[
	rnn/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	rnn/sub/xp
rnn/subSubrnn/sub/x:output:0rnn/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/subj
	rnn/mul_2Mulrnn/sub:z:0rnn/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_2o
	rnn/add_3AddV2rnn/mul_1:z:0rnn/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!rnn/TensorArrayV2_1/element_shapeШ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterц
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0rnn_readvariableop_resource"rnn_matmul_readvariableop_resource$rnn_matmul_1_readvariableop_resource^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *!
bodyR
rnn_while_body_199545*!
condR
rnn_while_cond_199544*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
	rnn/whileН
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeј
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2В
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permЕ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
rnn/transpose_1
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1/unstackЎ
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02"
 gru_cell_1/MatMul/ReadVariableOpИ
gru_cell_1/MatMulMatMul*tf_op_layer_ExpandDims/ExpandDims:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split/split_dimи
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/splitД
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOpІ
gru_cell_1/MatMul_1MatMulrnn/while:output:4*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul_1Ѕ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_2r
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Tanh
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0rnn/while:output:4*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_3Ѕ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulgru_cell_1/add_3:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddЙ
gru_cell_1_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_1/ReadVariableOp
gru_cell_1_1/unstackUnpack#gru_cell_1_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_1/unstackе
"gru_cell_1_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource!^gru_cell_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_1/MatMul/ReadVariableOpЌ
gru_cell_1_1/MatMulMatMuldense_2/BiasAdd:output:0*gru_cell_1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMulЇ
gru_cell_1_1/BiasAddBiasAddgru_cell_1_1/MatMul:product:0gru_cell_1_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAddj
gru_cell_1_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_1/Const
gru_cell_1_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_1/split/split_dimр
gru_cell_1_1/splitSplit%gru_cell_1_1/split/split_dim:output:0gru_cell_1_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/splitн
$gru_cell_1_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource#^gru_cell_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_1/MatMul_1/ReadVariableOpЎ
gru_cell_1_1/MatMul_1MatMulgru_cell_1/add_3:z:0,gru_cell_1_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMul_1­
gru_cell_1_1/BiasAdd_1BiasAddgru_cell_1_1/MatMul_1:product:0gru_cell_1_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAdd_1
gru_cell_1_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_1/Const_1
gru_cell_1_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_1/split_1/split_dim
gru_cell_1_1/split_1SplitVgru_cell_1_1/BiasAdd_1:output:0gru_cell_1_1/Const_1:output:0'gru_cell_1_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/split_1
gru_cell_1_1/addAddV2gru_cell_1_1/split:output:0gru_cell_1_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add
gru_cell_1_1/SigmoidSigmoidgru_cell_1_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid
gru_cell_1_1/add_1AddV2gru_cell_1_1/split:output:1gru_cell_1_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_1
gru_cell_1_1/Sigmoid_1Sigmoidgru_cell_1_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid_1
gru_cell_1_1/mulMulgru_cell_1_1/Sigmoid_1:y:0gru_cell_1_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul
gru_cell_1_1/add_2AddV2gru_cell_1_1/split:output:2gru_cell_1_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_2x
gru_cell_1_1/TanhTanhgru_cell_1_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Tanh
gru_cell_1_1/mul_1Mulgru_cell_1_1/Sigmoid:y:0gru_cell_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_1m
gru_cell_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_1/sub/x
gru_cell_1_1/subSubgru_cell_1_1/sub/x:output:0gru_cell_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/sub
gru_cell_1_1/mul_2Mulgru_cell_1_1/sub:z:0gru_cell_1_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_2
gru_cell_1_1/add_3AddV2gru_cell_1_1/mul_1:z:0gru_cell_1_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_3Ѕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulgru_cell_1_1/add_3:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddЛ
gru_cell_1_2/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_2/ReadVariableOp
gru_cell_1_2/unstackUnpack#gru_cell_1_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_2/unstackз
"gru_cell_1_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_2/MatMul/ReadVariableOpЌ
gru_cell_1_2/MatMulMatMuldense_3/BiasAdd:output:0*gru_cell_1_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMulЇ
gru_cell_1_2/BiasAddBiasAddgru_cell_1_2/MatMul:product:0gru_cell_1_2/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAddj
gru_cell_1_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_2/Const
gru_cell_1_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_2/split/split_dimр
gru_cell_1_2/splitSplit%gru_cell_1_2/split/split_dim:output:0gru_cell_1_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/splitп
$gru_cell_1_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_2/MatMul_1/ReadVariableOpА
gru_cell_1_2/MatMul_1MatMulgru_cell_1_1/add_3:z:0,gru_cell_1_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMul_1­
gru_cell_1_2/BiasAdd_1BiasAddgru_cell_1_2/MatMul_1:product:0gru_cell_1_2/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAdd_1
gru_cell_1_2/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_2/Const_1
gru_cell_1_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_2/split_1/split_dim
gru_cell_1_2/split_1SplitVgru_cell_1_2/BiasAdd_1:output:0gru_cell_1_2/Const_1:output:0'gru_cell_1_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/split_1
gru_cell_1_2/addAddV2gru_cell_1_2/split:output:0gru_cell_1_2/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add
gru_cell_1_2/SigmoidSigmoidgru_cell_1_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid
gru_cell_1_2/add_1AddV2gru_cell_1_2/split:output:1gru_cell_1_2/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_1
gru_cell_1_2/Sigmoid_1Sigmoidgru_cell_1_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid_1
gru_cell_1_2/mulMulgru_cell_1_2/Sigmoid_1:y:0gru_cell_1_2/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul
gru_cell_1_2/add_2AddV2gru_cell_1_2/split:output:2gru_cell_1_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_2x
gru_cell_1_2/TanhTanhgru_cell_1_2/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Tanh
gru_cell_1_2/mul_1Mulgru_cell_1_2/Sigmoid:y:0gru_cell_1_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_1m
gru_cell_1_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_2/sub/x
gru_cell_1_2/subSubgru_cell_1_2/sub/x:output:0gru_cell_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/sub
gru_cell_1_2/mul_2Mulgru_cell_1_2/sub:z:0gru_cell_1_2/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_2
gru_cell_1_2/add_3AddV2gru_cell_1_2/mul_1:z:0gru_cell_1_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_3Ѕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulgru_cell_1_2/add_3:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЄ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЁ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddЛ
gru_cell_1_3/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_2/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_3/ReadVariableOp
gru_cell_1_3/unstackUnpack#gru_cell_1_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_3/unstackз
"gru_cell_1_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_2/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_3/MatMul/ReadVariableOpЌ
gru_cell_1_3/MatMulMatMuldense_4/BiasAdd:output:0*gru_cell_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMulЇ
gru_cell_1_3/BiasAddBiasAddgru_cell_1_3/MatMul:product:0gru_cell_1_3/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAddj
gru_cell_1_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_3/Const
gru_cell_1_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_3/split/split_dimр
gru_cell_1_3/splitSplit%gru_cell_1_3/split/split_dim:output:0gru_cell_1_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/splitп
$gru_cell_1_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_2/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_3/MatMul_1/ReadVariableOpА
gru_cell_1_3/MatMul_1MatMulgru_cell_1_2/add_3:z:0,gru_cell_1_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMul_1­
gru_cell_1_3/BiasAdd_1BiasAddgru_cell_1_3/MatMul_1:product:0gru_cell_1_3/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAdd_1
gru_cell_1_3/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_3/Const_1
gru_cell_1_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_3/split_1/split_dim
gru_cell_1_3/split_1SplitVgru_cell_1_3/BiasAdd_1:output:0gru_cell_1_3/Const_1:output:0'gru_cell_1_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/split_1
gru_cell_1_3/addAddV2gru_cell_1_3/split:output:0gru_cell_1_3/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add
gru_cell_1_3/SigmoidSigmoidgru_cell_1_3/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid
gru_cell_1_3/add_1AddV2gru_cell_1_3/split:output:1gru_cell_1_3/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_1
gru_cell_1_3/Sigmoid_1Sigmoidgru_cell_1_3/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid_1
gru_cell_1_3/mulMulgru_cell_1_3/Sigmoid_1:y:0gru_cell_1_3/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul
gru_cell_1_3/add_2AddV2gru_cell_1_3/split:output:2gru_cell_1_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_2x
gru_cell_1_3/TanhTanhgru_cell_1_3/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Tanh
gru_cell_1_3/mul_1Mulgru_cell_1_3/Sigmoid:y:0gru_cell_1_2/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_1m
gru_cell_1_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_3/sub/x
gru_cell_1_3/subSubgru_cell_1_3/sub/x:output:0gru_cell_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/sub
gru_cell_1_3/mul_2Mulgru_cell_1_3/sub:z:0gru_cell_1_3/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_2
gru_cell_1_3/add_3AddV2gru_cell_1_3/mul_1:z:0gru_cell_1_3/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_3Ѕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulgru_cell_1_3/add_3:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddє
tf_op_layer_packed/packedPackdense_2/BiasAdd:output:0dense_3/BiasAdd:output:0dense_4/BiasAdd:output:0dense_5/BiasAdd:output:0*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
tf_op_layer_packed/packedЁ
$tf_op_layer_transpose/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$tf_op_layer_transpose/transpose/permч
tf_op_layer_transpose/transpose	Transpose"tf_op_layer_packed/packed:output:0-tf_op_layer_transpose/transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_transpose/transposeќ
IdentityIdentity#tf_op_layer_transpose/transpose:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp#^gru_cell_1_1/MatMul/ReadVariableOp%^gru_cell_1_1/MatMul_1/ReadVariableOp^gru_cell_1_1/ReadVariableOp#^gru_cell_1_2/MatMul/ReadVariableOp%^gru_cell_1_2/MatMul_1/ReadVariableOp^gru_cell_1_2/ReadVariableOp#^gru_cell_1_3/MatMul/ReadVariableOp%^gru_cell_1_3/MatMul_1/ReadVariableOp^gru_cell_1_3/ReadVariableOp^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp
^rnn/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2H
"gru_cell_1_1/MatMul/ReadVariableOp"gru_cell_1_1/MatMul/ReadVariableOp2L
$gru_cell_1_1/MatMul_1/ReadVariableOp$gru_cell_1_1/MatMul_1/ReadVariableOp2:
gru_cell_1_1/ReadVariableOpgru_cell_1_1/ReadVariableOp2H
"gru_cell_1_2/MatMul/ReadVariableOp"gru_cell_1_2/MatMul/ReadVariableOp2L
$gru_cell_1_2/MatMul_1/ReadVariableOp$gru_cell_1_2/MatMul_1/ReadVariableOp2:
gru_cell_1_2/ReadVariableOpgru_cell_1_2/ReadVariableOp2H
"gru_cell_1_3/MatMul/ReadVariableOp"gru_cell_1_3/MatMul/ReadVariableOp2L
$gru_cell_1_3/MatMul_1/ReadVariableOp$gru_cell_1_3/MatMul_1/ReadVariableOp2:
gru_cell_1_3/ReadVariableOpgru_cell_1_3/ReadVariableOp26
rnn/MatMul/ReadVariableOprnn/MatMul/ReadVariableOp2:
rnn/MatMul_1/ReadVariableOprnn/MatMul_1/ReadVariableOp2(
rnn/ReadVariableOprnn/ReadVariableOp2
	rnn/while	rnn/while:' #
!
_user_specified_name	input_1
ѓ
ш
while_cond_200699
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_200699___redundant_placeholder0.
*while_cond_200699___redundant_placeholder1.
*while_cond_200699___redundant_placeholder2.
*while_cond_200699___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
ъ
Ћ
D__inference_gru_cell_layer_call_and_return_conditional_losses_201734

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
д
X
:__inference_tf_op_layer_strided_slice_layer_call_fn_200142
inputs_0
identity
strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2
strided_slice/begin{
strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/end
strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/strides
strided_sliceStridedSliceinputs_0strided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicef
IdentityIdentitystrided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
ф
У
C__inference_seq2seq_layer_call_and_return_conditional_losses_199151
input_1
rnn_readvariableop_resource&
"rnn_matmul_readvariableop_resource(
$rnn_matmul_1_readvariableop_resource&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂ gru_cell_1/MatMul/ReadVariableOpЂ"gru_cell_1/MatMul_1/ReadVariableOpЂgru_cell_1/ReadVariableOpЂ"gru_cell_1_1/MatMul/ReadVariableOpЂ$gru_cell_1_1/MatMul_1/ReadVariableOpЂgru_cell_1_1/ReadVariableOpЂ"gru_cell_1_2/MatMul/ReadVariableOpЂ$gru_cell_1_2/MatMul_1/ReadVariableOpЂgru_cell_1_2/ReadVariableOpЂ"gru_cell_1_3/MatMul/ReadVariableOpЂ$gru_cell_1_3/MatMul_1/ReadVariableOpЂgru_cell_1_3/ReadVariableOpЂrnn/MatMul/ReadVariableOpЂrnn/MatMul_1/ReadVariableOpЂrnn/ReadVariableOpЂ	rnn/whileГ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2/
-tf_op_layer_strided_slice/strided_slice/beginЏ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2-
+tf_op_layer_strided_slice/strided_slice/endЗ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/tf_op_layer_strided_slice/strided_slice/strides
'tf_op_layer_strided_slice/strided_sliceStridedSliceinput_16tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2)
'tf_op_layer_strided_slice/strided_slice
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimї
!tf_op_layer_ExpandDims/ExpandDims
ExpandDims0tf_op_layer_strided_slice/strided_slice:output:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2#
!tf_op_layer_ExpandDims/ExpandDimsM
	rnn/ShapeShapeinput_1*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2њ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/zeros}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposeinput_1rnn/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
rnn/TensorArrayV2/element_shapeТ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ч
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
rnn/strided_slice_2
rnn/ReadVariableOpReadVariableOprnn_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/ReadVariableOpv
rnn/unstackUnpackrnn/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
rnn/unstack
rnn/MatMul/ReadVariableOpReadVariableOp"rnn_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/MatMul/ReadVariableOp

rnn/MatMulMatMulrnn/strided_slice_2:output:0!rnn/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

rnn/MatMul
rnn/BiasAddBiasAddrnn/MatMul:product:0rnn/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAddX
	rnn/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
	rnn/Constu
rnn/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split/split_dimМ
	rnn/splitSplitrnn/split/split_dim:output:0rnn/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
	rnn/split
rnn/MatMul_1/ReadVariableOpReadVariableOp$rnn_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
rnn/MatMul_1/ReadVariableOp
rnn/MatMul_1MatMulrnn/zeros:output:0#rnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/MatMul_1
rnn/BiasAdd_1BiasAddrnn/MatMul_1:product:0rnn/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAdd_1o
rnn/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
rnn/Const_1y
rnn/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split_1/split_dimч
rnn/split_1SplitVrnn/BiasAdd_1:output:0rnn/Const_1:output:0rnn/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
rnn/split_1w
rnn/addAddV2rnn/split:output:0rnn/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/addd
rnn/SigmoidSigmoidrnn/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid{
	rnn/add_1AddV2rnn/split:output:1rnn/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_1j
rnn/Sigmoid_1Sigmoidrnn/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid_1t
rnn/mulMulrnn/Sigmoid_1:y:0rnn/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/mulr
	rnn/add_2AddV2rnn/split:output:2rnn/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_2]
rnn/TanhTanhrnn/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

rnn/Tanht
	rnn/mul_1Mulrnn/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_1[
	rnn/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	rnn/sub/xp
rnn/subSubrnn/sub/x:output:0rnn/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/subj
	rnn/mul_2Mulrnn/sub:z:0rnn/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_2o
	rnn/add_3AddV2rnn/mul_1:z:0rnn/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!rnn/TensorArrayV2_1/element_shapeШ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterц
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0rnn_readvariableop_resource"rnn_matmul_readvariableop_resource$rnn_matmul_1_readvariableop_resource^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *!
bodyR
rnn_while_body_198626*!
condR
rnn_while_cond_198625*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
	rnn/whileН
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeј
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2В
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permЕ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
rnn/transpose_1
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1/unstackЎ
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02"
 gru_cell_1/MatMul/ReadVariableOpИ
gru_cell_1/MatMulMatMul*tf_op_layer_ExpandDims/ExpandDims:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split/split_dimи
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/splitД
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOpІ
gru_cell_1/MatMul_1MatMulrnn/while:output:4*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul_1Ѕ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_2r
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Tanh
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0rnn/while:output:4*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_3Ѕ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulgru_cell_1/add_3:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddЙ
gru_cell_1_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_1/ReadVariableOp
gru_cell_1_1/unstackUnpack#gru_cell_1_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_1/unstackе
"gru_cell_1_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource!^gru_cell_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_1/MatMul/ReadVariableOpЌ
gru_cell_1_1/MatMulMatMuldense_2/BiasAdd:output:0*gru_cell_1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMulЇ
gru_cell_1_1/BiasAddBiasAddgru_cell_1_1/MatMul:product:0gru_cell_1_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAddj
gru_cell_1_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_1/Const
gru_cell_1_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_1/split/split_dimр
gru_cell_1_1/splitSplit%gru_cell_1_1/split/split_dim:output:0gru_cell_1_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/splitн
$gru_cell_1_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource#^gru_cell_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_1/MatMul_1/ReadVariableOpЎ
gru_cell_1_1/MatMul_1MatMulgru_cell_1/add_3:z:0,gru_cell_1_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMul_1­
gru_cell_1_1/BiasAdd_1BiasAddgru_cell_1_1/MatMul_1:product:0gru_cell_1_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAdd_1
gru_cell_1_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_1/Const_1
gru_cell_1_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_1/split_1/split_dim
gru_cell_1_1/split_1SplitVgru_cell_1_1/BiasAdd_1:output:0gru_cell_1_1/Const_1:output:0'gru_cell_1_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/split_1
gru_cell_1_1/addAddV2gru_cell_1_1/split:output:0gru_cell_1_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add
gru_cell_1_1/SigmoidSigmoidgru_cell_1_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid
gru_cell_1_1/add_1AddV2gru_cell_1_1/split:output:1gru_cell_1_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_1
gru_cell_1_1/Sigmoid_1Sigmoidgru_cell_1_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid_1
gru_cell_1_1/mulMulgru_cell_1_1/Sigmoid_1:y:0gru_cell_1_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul
gru_cell_1_1/add_2AddV2gru_cell_1_1/split:output:2gru_cell_1_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_2x
gru_cell_1_1/TanhTanhgru_cell_1_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Tanh
gru_cell_1_1/mul_1Mulgru_cell_1_1/Sigmoid:y:0gru_cell_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_1m
gru_cell_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_1/sub/x
gru_cell_1_1/subSubgru_cell_1_1/sub/x:output:0gru_cell_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/sub
gru_cell_1_1/mul_2Mulgru_cell_1_1/sub:z:0gru_cell_1_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_2
gru_cell_1_1/add_3AddV2gru_cell_1_1/mul_1:z:0gru_cell_1_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_3Ѕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulgru_cell_1_1/add_3:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddЛ
gru_cell_1_2/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_2/ReadVariableOp
gru_cell_1_2/unstackUnpack#gru_cell_1_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_2/unstackз
"gru_cell_1_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_2/MatMul/ReadVariableOpЌ
gru_cell_1_2/MatMulMatMuldense_3/BiasAdd:output:0*gru_cell_1_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMulЇ
gru_cell_1_2/BiasAddBiasAddgru_cell_1_2/MatMul:product:0gru_cell_1_2/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAddj
gru_cell_1_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_2/Const
gru_cell_1_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_2/split/split_dimр
gru_cell_1_2/splitSplit%gru_cell_1_2/split/split_dim:output:0gru_cell_1_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/splitп
$gru_cell_1_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_2/MatMul_1/ReadVariableOpА
gru_cell_1_2/MatMul_1MatMulgru_cell_1_1/add_3:z:0,gru_cell_1_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMul_1­
gru_cell_1_2/BiasAdd_1BiasAddgru_cell_1_2/MatMul_1:product:0gru_cell_1_2/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAdd_1
gru_cell_1_2/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_2/Const_1
gru_cell_1_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_2/split_1/split_dim
gru_cell_1_2/split_1SplitVgru_cell_1_2/BiasAdd_1:output:0gru_cell_1_2/Const_1:output:0'gru_cell_1_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/split_1
gru_cell_1_2/addAddV2gru_cell_1_2/split:output:0gru_cell_1_2/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add
gru_cell_1_2/SigmoidSigmoidgru_cell_1_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid
gru_cell_1_2/add_1AddV2gru_cell_1_2/split:output:1gru_cell_1_2/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_1
gru_cell_1_2/Sigmoid_1Sigmoidgru_cell_1_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid_1
gru_cell_1_2/mulMulgru_cell_1_2/Sigmoid_1:y:0gru_cell_1_2/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul
gru_cell_1_2/add_2AddV2gru_cell_1_2/split:output:2gru_cell_1_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_2x
gru_cell_1_2/TanhTanhgru_cell_1_2/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Tanh
gru_cell_1_2/mul_1Mulgru_cell_1_2/Sigmoid:y:0gru_cell_1_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_1m
gru_cell_1_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_2/sub/x
gru_cell_1_2/subSubgru_cell_1_2/sub/x:output:0gru_cell_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/sub
gru_cell_1_2/mul_2Mulgru_cell_1_2/sub:z:0gru_cell_1_2/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_2
gru_cell_1_2/add_3AddV2gru_cell_1_2/mul_1:z:0gru_cell_1_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_3Ѕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulgru_cell_1_2/add_3:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЄ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЁ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddЛ
gru_cell_1_3/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_2/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_3/ReadVariableOp
gru_cell_1_3/unstackUnpack#gru_cell_1_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_3/unstackз
"gru_cell_1_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_2/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_3/MatMul/ReadVariableOpЌ
gru_cell_1_3/MatMulMatMuldense_4/BiasAdd:output:0*gru_cell_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMulЇ
gru_cell_1_3/BiasAddBiasAddgru_cell_1_3/MatMul:product:0gru_cell_1_3/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAddj
gru_cell_1_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_3/Const
gru_cell_1_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_3/split/split_dimр
gru_cell_1_3/splitSplit%gru_cell_1_3/split/split_dim:output:0gru_cell_1_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/splitп
$gru_cell_1_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_2/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_3/MatMul_1/ReadVariableOpА
gru_cell_1_3/MatMul_1MatMulgru_cell_1_2/add_3:z:0,gru_cell_1_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMul_1­
gru_cell_1_3/BiasAdd_1BiasAddgru_cell_1_3/MatMul_1:product:0gru_cell_1_3/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAdd_1
gru_cell_1_3/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_3/Const_1
gru_cell_1_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_3/split_1/split_dim
gru_cell_1_3/split_1SplitVgru_cell_1_3/BiasAdd_1:output:0gru_cell_1_3/Const_1:output:0'gru_cell_1_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/split_1
gru_cell_1_3/addAddV2gru_cell_1_3/split:output:0gru_cell_1_3/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add
gru_cell_1_3/SigmoidSigmoidgru_cell_1_3/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid
gru_cell_1_3/add_1AddV2gru_cell_1_3/split:output:1gru_cell_1_3/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_1
gru_cell_1_3/Sigmoid_1Sigmoidgru_cell_1_3/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid_1
gru_cell_1_3/mulMulgru_cell_1_3/Sigmoid_1:y:0gru_cell_1_3/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul
gru_cell_1_3/add_2AddV2gru_cell_1_3/split:output:2gru_cell_1_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_2x
gru_cell_1_3/TanhTanhgru_cell_1_3/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Tanh
gru_cell_1_3/mul_1Mulgru_cell_1_3/Sigmoid:y:0gru_cell_1_2/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_1m
gru_cell_1_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_3/sub/x
gru_cell_1_3/subSubgru_cell_1_3/sub/x:output:0gru_cell_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/sub
gru_cell_1_3/mul_2Mulgru_cell_1_3/sub:z:0gru_cell_1_3/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_2
gru_cell_1_3/add_3AddV2gru_cell_1_3/mul_1:z:0gru_cell_1_3/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_3Ѕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulgru_cell_1_3/add_3:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddє
tf_op_layer_packed/packedPackdense_2/BiasAdd:output:0dense_3/BiasAdd:output:0dense_4/BiasAdd:output:0dense_5/BiasAdd:output:0*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
tf_op_layer_packed/packedЁ
$tf_op_layer_transpose/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$tf_op_layer_transpose/transpose/permч
tf_op_layer_transpose/transpose	Transpose"tf_op_layer_packed/packed:output:0-tf_op_layer_transpose/transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_transpose/transposeќ
IdentityIdentity#tf_op_layer_transpose/transpose:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp#^gru_cell_1_1/MatMul/ReadVariableOp%^gru_cell_1_1/MatMul_1/ReadVariableOp^gru_cell_1_1/ReadVariableOp#^gru_cell_1_2/MatMul/ReadVariableOp%^gru_cell_1_2/MatMul_1/ReadVariableOp^gru_cell_1_2/ReadVariableOp#^gru_cell_1_3/MatMul/ReadVariableOp%^gru_cell_1_3/MatMul_1/ReadVariableOp^gru_cell_1_3/ReadVariableOp^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp
^rnn/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2H
"gru_cell_1_1/MatMul/ReadVariableOp"gru_cell_1_1/MatMul/ReadVariableOp2L
$gru_cell_1_1/MatMul_1/ReadVariableOp$gru_cell_1_1/MatMul_1/ReadVariableOp2:
gru_cell_1_1/ReadVariableOpgru_cell_1_1/ReadVariableOp2H
"gru_cell_1_2/MatMul/ReadVariableOp"gru_cell_1_2/MatMul/ReadVariableOp2L
$gru_cell_1_2/MatMul_1/ReadVariableOp$gru_cell_1_2/MatMul_1/ReadVariableOp2:
gru_cell_1_2/ReadVariableOpgru_cell_1_2/ReadVariableOp2H
"gru_cell_1_3/MatMul/ReadVariableOp"gru_cell_1_3/MatMul/ReadVariableOp2L
$gru_cell_1_3/MatMul_1/ReadVariableOp$gru_cell_1_3/MatMul_1/ReadVariableOp2:
gru_cell_1_3/ReadVariableOpgru_cell_1_3/ReadVariableOp26
rnn/MatMul/ReadVariableOprnn/MatMul/ReadVariableOp2:
rnn/MatMul_1/ReadVariableOprnn/MatMul_1/ReadVariableOp2(
rnn/ReadVariableOprnn/ReadVariableOp2
	rnn/while	rnn/while:' #
!
_user_specified_name	input_1
ф
У
C__inference_seq2seq_layer_call_and_return_conditional_losses_199469
input_1
rnn_readvariableop_resource&
"rnn_matmul_readvariableop_resource(
$rnn_matmul_1_readvariableop_resource&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂ gru_cell_1/MatMul/ReadVariableOpЂ"gru_cell_1/MatMul_1/ReadVariableOpЂgru_cell_1/ReadVariableOpЂ"gru_cell_1_1/MatMul/ReadVariableOpЂ$gru_cell_1_1/MatMul_1/ReadVariableOpЂgru_cell_1_1/ReadVariableOpЂ"gru_cell_1_2/MatMul/ReadVariableOpЂ$gru_cell_1_2/MatMul_1/ReadVariableOpЂgru_cell_1_2/ReadVariableOpЂ"gru_cell_1_3/MatMul/ReadVariableOpЂ$gru_cell_1_3/MatMul_1/ReadVariableOpЂgru_cell_1_3/ReadVariableOpЂrnn/MatMul/ReadVariableOpЂrnn/MatMul_1/ReadVariableOpЂrnn/ReadVariableOpЂ	rnn/whileГ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2/
-tf_op_layer_strided_slice/strided_slice/beginЏ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2-
+tf_op_layer_strided_slice/strided_slice/endЗ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/tf_op_layer_strided_slice/strided_slice/strides
'tf_op_layer_strided_slice/strided_sliceStridedSliceinput_16tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2)
'tf_op_layer_strided_slice/strided_slice
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimї
!tf_op_layer_ExpandDims/ExpandDims
ExpandDims0tf_op_layer_strided_slice/strided_slice:output:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2#
!tf_op_layer_ExpandDims/ExpandDimsM
	rnn/ShapeShapeinput_1*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2њ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/zeros}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposeinput_1rnn/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
rnn/TensorArrayV2/element_shapeТ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ч
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
rnn/strided_slice_2
rnn/ReadVariableOpReadVariableOprnn_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/ReadVariableOpv
rnn/unstackUnpackrnn/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
rnn/unstack
rnn/MatMul/ReadVariableOpReadVariableOp"rnn_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/MatMul/ReadVariableOp

rnn/MatMulMatMulrnn/strided_slice_2:output:0!rnn/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

rnn/MatMul
rnn/BiasAddBiasAddrnn/MatMul:product:0rnn/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAddX
	rnn/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
	rnn/Constu
rnn/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split/split_dimМ
	rnn/splitSplitrnn/split/split_dim:output:0rnn/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
	rnn/split
rnn/MatMul_1/ReadVariableOpReadVariableOp$rnn_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
rnn/MatMul_1/ReadVariableOp
rnn/MatMul_1MatMulrnn/zeros:output:0#rnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/MatMul_1
rnn/BiasAdd_1BiasAddrnn/MatMul_1:product:0rnn/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAdd_1o
rnn/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
rnn/Const_1y
rnn/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split_1/split_dimч
rnn/split_1SplitVrnn/BiasAdd_1:output:0rnn/Const_1:output:0rnn/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
rnn/split_1w
rnn/addAddV2rnn/split:output:0rnn/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/addd
rnn/SigmoidSigmoidrnn/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid{
	rnn/add_1AddV2rnn/split:output:1rnn/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_1j
rnn/Sigmoid_1Sigmoidrnn/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid_1t
rnn/mulMulrnn/Sigmoid_1:y:0rnn/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/mulr
	rnn/add_2AddV2rnn/split:output:2rnn/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_2]
rnn/TanhTanhrnn/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

rnn/Tanht
	rnn/mul_1Mulrnn/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_1[
	rnn/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	rnn/sub/xp
rnn/subSubrnn/sub/x:output:0rnn/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/subj
	rnn/mul_2Mulrnn/sub:z:0rnn/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_2o
	rnn/add_3AddV2rnn/mul_1:z:0rnn/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!rnn/TensorArrayV2_1/element_shapeШ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterц
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0rnn_readvariableop_resource"rnn_matmul_readvariableop_resource$rnn_matmul_1_readvariableop_resource^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *!
bodyR
rnn_while_body_199226*!
condR
rnn_while_cond_199225*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
	rnn/whileН
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeј
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2В
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permЕ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
rnn/transpose_1
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1/unstackЎ
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02"
 gru_cell_1/MatMul/ReadVariableOpИ
gru_cell_1/MatMulMatMul*tf_op_layer_ExpandDims/ExpandDims:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split/split_dimи
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/splitД
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOpІ
gru_cell_1/MatMul_1MatMulrnn/while:output:4*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul_1Ѕ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_2r
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Tanh
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0rnn/while:output:4*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_3Ѕ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulgru_cell_1/add_3:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddЙ
gru_cell_1_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_1/ReadVariableOp
gru_cell_1_1/unstackUnpack#gru_cell_1_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_1/unstackе
"gru_cell_1_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource!^gru_cell_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_1/MatMul/ReadVariableOpЌ
gru_cell_1_1/MatMulMatMuldense_2/BiasAdd:output:0*gru_cell_1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMulЇ
gru_cell_1_1/BiasAddBiasAddgru_cell_1_1/MatMul:product:0gru_cell_1_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAddj
gru_cell_1_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_1/Const
gru_cell_1_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_1/split/split_dimр
gru_cell_1_1/splitSplit%gru_cell_1_1/split/split_dim:output:0gru_cell_1_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/splitн
$gru_cell_1_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource#^gru_cell_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_1/MatMul_1/ReadVariableOpЎ
gru_cell_1_1/MatMul_1MatMulgru_cell_1/add_3:z:0,gru_cell_1_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMul_1­
gru_cell_1_1/BiasAdd_1BiasAddgru_cell_1_1/MatMul_1:product:0gru_cell_1_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAdd_1
gru_cell_1_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_1/Const_1
gru_cell_1_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_1/split_1/split_dim
gru_cell_1_1/split_1SplitVgru_cell_1_1/BiasAdd_1:output:0gru_cell_1_1/Const_1:output:0'gru_cell_1_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/split_1
gru_cell_1_1/addAddV2gru_cell_1_1/split:output:0gru_cell_1_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add
gru_cell_1_1/SigmoidSigmoidgru_cell_1_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid
gru_cell_1_1/add_1AddV2gru_cell_1_1/split:output:1gru_cell_1_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_1
gru_cell_1_1/Sigmoid_1Sigmoidgru_cell_1_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid_1
gru_cell_1_1/mulMulgru_cell_1_1/Sigmoid_1:y:0gru_cell_1_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul
gru_cell_1_1/add_2AddV2gru_cell_1_1/split:output:2gru_cell_1_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_2x
gru_cell_1_1/TanhTanhgru_cell_1_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Tanh
gru_cell_1_1/mul_1Mulgru_cell_1_1/Sigmoid:y:0gru_cell_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_1m
gru_cell_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_1/sub/x
gru_cell_1_1/subSubgru_cell_1_1/sub/x:output:0gru_cell_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/sub
gru_cell_1_1/mul_2Mulgru_cell_1_1/sub:z:0gru_cell_1_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_2
gru_cell_1_1/add_3AddV2gru_cell_1_1/mul_1:z:0gru_cell_1_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_3Ѕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulgru_cell_1_1/add_3:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddЛ
gru_cell_1_2/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_2/ReadVariableOp
gru_cell_1_2/unstackUnpack#gru_cell_1_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_2/unstackз
"gru_cell_1_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_2/MatMul/ReadVariableOpЌ
gru_cell_1_2/MatMulMatMuldense_3/BiasAdd:output:0*gru_cell_1_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMulЇ
gru_cell_1_2/BiasAddBiasAddgru_cell_1_2/MatMul:product:0gru_cell_1_2/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAddj
gru_cell_1_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_2/Const
gru_cell_1_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_2/split/split_dimр
gru_cell_1_2/splitSplit%gru_cell_1_2/split/split_dim:output:0gru_cell_1_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/splitп
$gru_cell_1_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_2/MatMul_1/ReadVariableOpА
gru_cell_1_2/MatMul_1MatMulgru_cell_1_1/add_3:z:0,gru_cell_1_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMul_1­
gru_cell_1_2/BiasAdd_1BiasAddgru_cell_1_2/MatMul_1:product:0gru_cell_1_2/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAdd_1
gru_cell_1_2/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_2/Const_1
gru_cell_1_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_2/split_1/split_dim
gru_cell_1_2/split_1SplitVgru_cell_1_2/BiasAdd_1:output:0gru_cell_1_2/Const_1:output:0'gru_cell_1_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/split_1
gru_cell_1_2/addAddV2gru_cell_1_2/split:output:0gru_cell_1_2/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add
gru_cell_1_2/SigmoidSigmoidgru_cell_1_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid
gru_cell_1_2/add_1AddV2gru_cell_1_2/split:output:1gru_cell_1_2/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_1
gru_cell_1_2/Sigmoid_1Sigmoidgru_cell_1_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid_1
gru_cell_1_2/mulMulgru_cell_1_2/Sigmoid_1:y:0gru_cell_1_2/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul
gru_cell_1_2/add_2AddV2gru_cell_1_2/split:output:2gru_cell_1_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_2x
gru_cell_1_2/TanhTanhgru_cell_1_2/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Tanh
gru_cell_1_2/mul_1Mulgru_cell_1_2/Sigmoid:y:0gru_cell_1_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_1m
gru_cell_1_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_2/sub/x
gru_cell_1_2/subSubgru_cell_1_2/sub/x:output:0gru_cell_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/sub
gru_cell_1_2/mul_2Mulgru_cell_1_2/sub:z:0gru_cell_1_2/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_2
gru_cell_1_2/add_3AddV2gru_cell_1_2/mul_1:z:0gru_cell_1_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_3Ѕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulgru_cell_1_2/add_3:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЄ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЁ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddЛ
gru_cell_1_3/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_2/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_3/ReadVariableOp
gru_cell_1_3/unstackUnpack#gru_cell_1_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_3/unstackз
"gru_cell_1_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_2/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_3/MatMul/ReadVariableOpЌ
gru_cell_1_3/MatMulMatMuldense_4/BiasAdd:output:0*gru_cell_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMulЇ
gru_cell_1_3/BiasAddBiasAddgru_cell_1_3/MatMul:product:0gru_cell_1_3/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAddj
gru_cell_1_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_3/Const
gru_cell_1_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_3/split/split_dimр
gru_cell_1_3/splitSplit%gru_cell_1_3/split/split_dim:output:0gru_cell_1_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/splitп
$gru_cell_1_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_2/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_3/MatMul_1/ReadVariableOpА
gru_cell_1_3/MatMul_1MatMulgru_cell_1_2/add_3:z:0,gru_cell_1_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMul_1­
gru_cell_1_3/BiasAdd_1BiasAddgru_cell_1_3/MatMul_1:product:0gru_cell_1_3/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAdd_1
gru_cell_1_3/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_3/Const_1
gru_cell_1_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_3/split_1/split_dim
gru_cell_1_3/split_1SplitVgru_cell_1_3/BiasAdd_1:output:0gru_cell_1_3/Const_1:output:0'gru_cell_1_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/split_1
gru_cell_1_3/addAddV2gru_cell_1_3/split:output:0gru_cell_1_3/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add
gru_cell_1_3/SigmoidSigmoidgru_cell_1_3/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid
gru_cell_1_3/add_1AddV2gru_cell_1_3/split:output:1gru_cell_1_3/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_1
gru_cell_1_3/Sigmoid_1Sigmoidgru_cell_1_3/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid_1
gru_cell_1_3/mulMulgru_cell_1_3/Sigmoid_1:y:0gru_cell_1_3/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul
gru_cell_1_3/add_2AddV2gru_cell_1_3/split:output:2gru_cell_1_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_2x
gru_cell_1_3/TanhTanhgru_cell_1_3/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Tanh
gru_cell_1_3/mul_1Mulgru_cell_1_3/Sigmoid:y:0gru_cell_1_2/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_1m
gru_cell_1_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_3/sub/x
gru_cell_1_3/subSubgru_cell_1_3/sub/x:output:0gru_cell_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/sub
gru_cell_1_3/mul_2Mulgru_cell_1_3/sub:z:0gru_cell_1_3/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_2
gru_cell_1_3/add_3AddV2gru_cell_1_3/mul_1:z:0gru_cell_1_3/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_3Ѕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulgru_cell_1_3/add_3:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddє
tf_op_layer_packed/packedPackdense_2/BiasAdd:output:0dense_3/BiasAdd:output:0dense_4/BiasAdd:output:0dense_5/BiasAdd:output:0*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
tf_op_layer_packed/packedЁ
$tf_op_layer_transpose/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$tf_op_layer_transpose/transpose/permч
tf_op_layer_transpose/transpose	Transpose"tf_op_layer_packed/packed:output:0-tf_op_layer_transpose/transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_transpose/transposeќ
IdentityIdentity#tf_op_layer_transpose/transpose:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp#^gru_cell_1_1/MatMul/ReadVariableOp%^gru_cell_1_1/MatMul_1/ReadVariableOp^gru_cell_1_1/ReadVariableOp#^gru_cell_1_2/MatMul/ReadVariableOp%^gru_cell_1_2/MatMul_1/ReadVariableOp^gru_cell_1_2/ReadVariableOp#^gru_cell_1_3/MatMul/ReadVariableOp%^gru_cell_1_3/MatMul_1/ReadVariableOp^gru_cell_1_3/ReadVariableOp^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp
^rnn/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2H
"gru_cell_1_1/MatMul/ReadVariableOp"gru_cell_1_1/MatMul/ReadVariableOp2L
$gru_cell_1_1/MatMul_1/ReadVariableOp$gru_cell_1_1/MatMul_1/ReadVariableOp2:
gru_cell_1_1/ReadVariableOpgru_cell_1_1/ReadVariableOp2H
"gru_cell_1_2/MatMul/ReadVariableOp"gru_cell_1_2/MatMul/ReadVariableOp2L
$gru_cell_1_2/MatMul_1/ReadVariableOp$gru_cell_1_2/MatMul_1/ReadVariableOp2:
gru_cell_1_2/ReadVariableOpgru_cell_1_2/ReadVariableOp2H
"gru_cell_1_3/MatMul/ReadVariableOp"gru_cell_1_3/MatMul/ReadVariableOp2L
$gru_cell_1_3/MatMul_1/ReadVariableOp$gru_cell_1_3/MatMul_1/ReadVariableOp2:
gru_cell_1_3/ReadVariableOpgru_cell_1_3/ReadVariableOp26
rnn/MatMul/ReadVariableOprnn/MatMul/ReadVariableOp2:
rnn/MatMul_1/ReadVariableOprnn/MatMul_1/ReadVariableOp2(
rnn/ReadVariableOprnn/ReadVariableOp2
	rnn/while	rnn/while:' #
!
_user_specified_name	input_1
ЉR

$__inference_rnn_layer_call_fn_201426
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_201336*
condR
while_cond_201335*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЎ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2
transpose_1Л
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapes.
,:џџџџџџџџџџџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
Я

)__inference_gru_cell_layer_call_fn_201854

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Я
л
$__inference_signature_wrapper_200126
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_1970822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
R
 
?__inference_rnn_layer_call_and_return_conditional_losses_200472

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_200382*
condR
while_cond_200381*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1В
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*+
_output_shapes
:џџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
ц2

seq2seq_rnn_while_body_196839"
seq2seq_rnn_while_loop_counter(
$seq2seq_rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2!
seq2seq_rnn_strided_slice_1_0]
Ytensorarrayv2read_tensorlistgetitem_seq2seq_rnn_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
seq2seq_rnn_strided_slice_1[
Wtensorarrayv2read_tensorlistgetitem_seq2seq_rnn_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeС
#TensorArrayV2Read/TensorListGetItemTensorListGetItemYtensorarrayv2read_tensorlistgetitem_seq2seq_rnn_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/yj
add_5AddV2seq2seq_rnn_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЎ

Identity_1Identity$seq2seq_rnn_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"<
seq2seq_rnn_strided_slice_1seq2seq_rnn_strided_slice_1_0"Д
Wtensorarrayv2read_tensorlistgetitem_seq2seq_rnn_tensorarrayunstack_tensorlistfromtensorYtensorarrayv2read_tensorlistgetitem_seq2seq_rnn_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp


rnn_while_cond_198625
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_rnn_strided_slice_12
.rnn_while_cond_198625___redundant_placeholder02
.rnn_while_cond_198625___redundant_placeholder12
.rnn_while_cond_198625___redundant_placeholder22
.rnn_while_cond_198625___redundant_placeholder3
identity
\
LessLessplaceholderless_rnn_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
і1
д
rnn_while_body_199863
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
rnn_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
rnn_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЙ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/yb
add_5AddV2rnn_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityІ

Identity_1Identityrnn_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0",
rnn_strided_slice_1rnn_strided_slice_1_0"Є
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
R
 
?__inference_rnn_layer_call_and_return_conditional_losses_200313

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_200223*
condR
while_cond_200222*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1В
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*+
_output_shapes
:џџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
ь
­
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201466

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
О1
И
while_body_200859
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
я
s
U__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_200134
inputs_0
identity
strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2
strided_slice/begin{
strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/end
strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/strides
strided_sliceStridedSliceinputs_0strided_slice/begin:output:0strided_slice/end:output:0strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicef
IdentityIdentitystrided_slice:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
ѓ
ш
while_cond_200540
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_200540___redundant_placeholder0.
*while_cond_200540___redundant_placeholder1.
*while_cond_200540___redundant_placeholder2.
*while_cond_200540___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
ѓ
ш
while_cond_201335
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_201335___redundant_placeholder0.
*while_cond_201335___redundant_placeholder1.
*while_cond_201335___redundant_placeholder2.
*while_cond_201335___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
О1
И
while_body_201336
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
ѓ
ш
while_cond_201176
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_201176___redundant_placeholder0.
*while_cond_201176___redundant_placeholder1.
*while_cond_201176___redundant_placeholder2.
*while_cond_201176___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
З
T
6__inference_tf_op_layer_transpose_layer_call_fn_201694
inputs_0
identityu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
	transposee
IdentityIdentitytranspose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0
і1
д
rnn_while_body_198626
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
rnn_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
rnn_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЙ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/yb
add_5AddV2rnn_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityІ

Identity_1Identityrnn_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0",
rnn_strided_slice_1rnn_strided_slice_1_0"Є
Otensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
ь
­
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201506

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

IdentityЄ

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:џџџџџџџџџ:џџџџџџџџџ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0

{
3__inference_tf_op_layer_packed_layer_call_fn_201682
inputs_0
inputs_1
inputs_2
inputs_3
identity
packedPackinputs_0inputs_1inputs_2inputs_3*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
packedg
IdentityIdentitypacked:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1:($
"
_user_specified_name
inputs/2:($
"
_user_specified_name
inputs/3
ѓ
ш
while_cond_200222
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_200222___redundant_placeholder0.
*while_cond_200222___redundant_placeholder1.
*while_cond_200222___redundant_placeholder2.
*while_cond_200222___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::


rnn_while_cond_199862
rnn_while_loop_counter 
rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_rnn_strided_slice_12
.rnn_while_cond_199862___redundant_placeholder02
.rnn_while_cond_199862___redundant_placeholder12
.rnn_while_cond_199862___redundant_placeholder22
.rnn_while_cond_199862___redundant_placeholder3
identity
\
LessLessplaceholderless_rnn_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
Щ
Ј
(__inference_seq2seq_layer_call_fn_200106
input_1
rnn_readvariableop_resource&
"rnn_matmul_readvariableop_resource(
$rnn_matmul_1_readvariableop_resource&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpЂdense_4/BiasAdd/ReadVariableOpЂdense_4/MatMul/ReadVariableOpЂdense_5/BiasAdd/ReadVariableOpЂdense_5/MatMul/ReadVariableOpЂ gru_cell_1/MatMul/ReadVariableOpЂ"gru_cell_1/MatMul_1/ReadVariableOpЂgru_cell_1/ReadVariableOpЂ"gru_cell_1_1/MatMul/ReadVariableOpЂ$gru_cell_1_1/MatMul_1/ReadVariableOpЂgru_cell_1_1/ReadVariableOpЂ"gru_cell_1_2/MatMul/ReadVariableOpЂ$gru_cell_1_2/MatMul_1/ReadVariableOpЂgru_cell_1_2/ReadVariableOpЂ"gru_cell_1_3/MatMul/ReadVariableOpЂ$gru_cell_1_3/MatMul_1/ReadVariableOpЂgru_cell_1_3/ReadVariableOpЂrnn/MatMul/ReadVariableOpЂrnn/MatMul_1/ReadVariableOpЂrnn/ReadVariableOpЂ	rnn/whileГ
-tf_op_layer_strided_slice/strided_slice/beginConst*
_output_shapes
:*
dtype0*!
valueB"    џџџџ    2/
-tf_op_layer_strided_slice/strided_slice/beginЏ
+tf_op_layer_strided_slice/strided_slice/endConst*
_output_shapes
:*
dtype0*!
valueB"           2-
+tf_op_layer_strided_slice/strided_slice/endЗ
/tf_op_layer_strided_slice/strided_slice/stridesConst*
_output_shapes
:*
dtype0*!
valueB"         21
/tf_op_layer_strided_slice/strided_slice/strides
'tf_op_layer_strided_slice/strided_sliceStridedSliceinput_16tf_op_layer_strided_slice/strided_slice/begin:output:04tf_op_layer_strided_slice/strided_slice/end:output:08tf_op_layer_strided_slice/strided_slice/strides:output:0*
Index0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask*
shrink_axis_mask2)
'tf_op_layer_strided_slice/strided_slice
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimї
!tf_op_layer_ExpandDims/ExpandDims
ExpandDims0tf_op_layer_strided_slice/strided_slice:output:0.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2#
!tf_op_layer_ExpandDims/ExpandDimsM
	rnn/ShapeShapeinput_1*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2њ
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/zeros}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposeinput_1rnn/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2!
rnn/TensorArrayV2/element_shapeТ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ч
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
rnn/strided_slice_2
rnn/ReadVariableOpReadVariableOprnn_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/ReadVariableOpv
rnn/unstackUnpackrnn/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
rnn/unstack
rnn/MatMul/ReadVariableOpReadVariableOp"rnn_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02
rnn/MatMul/ReadVariableOp

rnn/MatMulMatMulrnn/strided_slice_2:output:0!rnn/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

rnn/MatMul
rnn/BiasAddBiasAddrnn/MatMul:product:0rnn/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAddX
	rnn/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
	rnn/Constu
rnn/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split/split_dimМ
	rnn/splitSplitrnn/split/split_dim:output:0rnn/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
	rnn/split
rnn/MatMul_1/ReadVariableOpReadVariableOp$rnn_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
rnn/MatMul_1/ReadVariableOp
rnn/MatMul_1MatMulrnn/zeros:output:0#rnn/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/MatMul_1
rnn/BiasAdd_1BiasAddrnn/MatMul_1:product:0rnn/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
rnn/BiasAdd_1o
rnn/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
rnn/Const_1y
rnn/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/split_1/split_dimч
rnn/split_1SplitVrnn/BiasAdd_1:output:0rnn/Const_1:output:0rnn/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
rnn/split_1w
rnn/addAddV2rnn/split:output:0rnn/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/addd
rnn/SigmoidSigmoidrnn/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid{
	rnn/add_1AddV2rnn/split:output:1rnn/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_1j
rnn/Sigmoid_1Sigmoidrnn/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
rnn/Sigmoid_1t
rnn/mulMulrnn/Sigmoid_1:y:0rnn/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/mulr
	rnn/add_2AddV2rnn/split:output:2rnn/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_2]
rnn/TanhTanhrnn/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2

rnn/Tanht
	rnn/mul_1Mulrnn/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_1[
	rnn/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	rnn/sub/xp
rnn/subSubrnn/sub/x:output:0rnn/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
rnn/subj
	rnn/mul_2Mulrnn/sub:z:0rnn/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/mul_2o
	rnn/add_3AddV2rnn/mul_1:z:0rnn/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	rnn/add_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2#
!rnn/TensorArrayV2_1/element_shapeШ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterц
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0rnn_readvariableop_resource"rnn_matmul_readvariableop_resource$rnn_matmul_1_readvariableop_resource^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *!
bodyR
rnn_while_body_199863*!
condR
rnn_while_cond_199862*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
	rnn/whileН
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeј
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2В
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permЕ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
rnn/transpose_1
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1/unstackЎ
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02"
 gru_cell_1/MatMul/ReadVariableOpИ
gru_cell_1/MatMulMatMul*tf_op_layer_ExpandDims/ExpandDims:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split/split_dimи
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/splitД
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOpІ
gru_cell_1/MatMul_1MatMulrnn/while:output:4*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/MatMul_1Ѕ
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_2r
gru_cell_1/TanhTanhgru_cell_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/Tanh
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0rnn/while:output:4*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1/add_3Ѕ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulgru_cell_1/add_3:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddЙ
gru_cell_1_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_1/ReadVariableOp
gru_cell_1_1/unstackUnpack#gru_cell_1_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_1/unstackе
"gru_cell_1_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource!^gru_cell_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_1/MatMul/ReadVariableOpЌ
gru_cell_1_1/MatMulMatMuldense_2/BiasAdd:output:0*gru_cell_1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMulЇ
gru_cell_1_1/BiasAddBiasAddgru_cell_1_1/MatMul:product:0gru_cell_1_1/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAddj
gru_cell_1_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_1/Const
gru_cell_1_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_1/split/split_dimр
gru_cell_1_1/splitSplit%gru_cell_1_1/split/split_dim:output:0gru_cell_1_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/splitн
$gru_cell_1_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource#^gru_cell_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_1/MatMul_1/ReadVariableOpЎ
gru_cell_1_1/MatMul_1MatMulgru_cell_1/add_3:z:0,gru_cell_1_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/MatMul_1­
gru_cell_1_1/BiasAdd_1BiasAddgru_cell_1_1/MatMul_1:product:0gru_cell_1_1/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_1/BiasAdd_1
gru_cell_1_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_1/Const_1
gru_cell_1_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_1/split_1/split_dim
gru_cell_1_1/split_1SplitVgru_cell_1_1/BiasAdd_1:output:0gru_cell_1_1/Const_1:output:0'gru_cell_1_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_1/split_1
gru_cell_1_1/addAddV2gru_cell_1_1/split:output:0gru_cell_1_1/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add
gru_cell_1_1/SigmoidSigmoidgru_cell_1_1/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid
gru_cell_1_1/add_1AddV2gru_cell_1_1/split:output:1gru_cell_1_1/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_1
gru_cell_1_1/Sigmoid_1Sigmoidgru_cell_1_1/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Sigmoid_1
gru_cell_1_1/mulMulgru_cell_1_1/Sigmoid_1:y:0gru_cell_1_1/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul
gru_cell_1_1/add_2AddV2gru_cell_1_1/split:output:2gru_cell_1_1/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_2x
gru_cell_1_1/TanhTanhgru_cell_1_1/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/Tanh
gru_cell_1_1/mul_1Mulgru_cell_1_1/Sigmoid:y:0gru_cell_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_1m
gru_cell_1_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_1/sub/x
gru_cell_1_1/subSubgru_cell_1_1/sub/x:output:0gru_cell_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/sub
gru_cell_1_1/mul_2Mulgru_cell_1_1/sub:z:0gru_cell_1_1/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/mul_2
gru_cell_1_1/add_3AddV2gru_cell_1_1/mul_1:z:0gru_cell_1_1/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_1/add_3Ѕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulgru_cell_1_1/add_3:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddЛ
gru_cell_1_2/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_1/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_2/ReadVariableOp
gru_cell_1_2/unstackUnpack#gru_cell_1_2/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_2/unstackз
"gru_cell_1_2/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_1/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_2/MatMul/ReadVariableOpЌ
gru_cell_1_2/MatMulMatMuldense_3/BiasAdd:output:0*gru_cell_1_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMulЇ
gru_cell_1_2/BiasAddBiasAddgru_cell_1_2/MatMul:product:0gru_cell_1_2/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAddj
gru_cell_1_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_2/Const
gru_cell_1_2/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_2/split/split_dimр
gru_cell_1_2/splitSplit%gru_cell_1_2/split/split_dim:output:0gru_cell_1_2/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/splitп
$gru_cell_1_2/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_1/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_2/MatMul_1/ReadVariableOpА
gru_cell_1_2/MatMul_1MatMulgru_cell_1_1/add_3:z:0,gru_cell_1_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/MatMul_1­
gru_cell_1_2/BiasAdd_1BiasAddgru_cell_1_2/MatMul_1:product:0gru_cell_1_2/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_2/BiasAdd_1
gru_cell_1_2/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_2/Const_1
gru_cell_1_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_2/split_1/split_dim
gru_cell_1_2/split_1SplitVgru_cell_1_2/BiasAdd_1:output:0gru_cell_1_2/Const_1:output:0'gru_cell_1_2/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_2/split_1
gru_cell_1_2/addAddV2gru_cell_1_2/split:output:0gru_cell_1_2/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add
gru_cell_1_2/SigmoidSigmoidgru_cell_1_2/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid
gru_cell_1_2/add_1AddV2gru_cell_1_2/split:output:1gru_cell_1_2/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_1
gru_cell_1_2/Sigmoid_1Sigmoidgru_cell_1_2/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Sigmoid_1
gru_cell_1_2/mulMulgru_cell_1_2/Sigmoid_1:y:0gru_cell_1_2/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul
gru_cell_1_2/add_2AddV2gru_cell_1_2/split:output:2gru_cell_1_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_2x
gru_cell_1_2/TanhTanhgru_cell_1_2/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/Tanh
gru_cell_1_2/mul_1Mulgru_cell_1_2/Sigmoid:y:0gru_cell_1_1/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_1m
gru_cell_1_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_2/sub/x
gru_cell_1_2/subSubgru_cell_1_2/sub/x:output:0gru_cell_1_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/sub
gru_cell_1_2/mul_2Mulgru_cell_1_2/sub:z:0gru_cell_1_2/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/mul_2
gru_cell_1_2/add_3AddV2gru_cell_1_2/mul_1:z:0gru_cell_1_2/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_2/add_3Ѕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMulgru_cell_1_2/add_3:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/MatMulЄ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЁ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_4/BiasAddЛ
gru_cell_1_3/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource^gru_cell_1_2/ReadVariableOp*
_output_shapes

:`*
dtype02
gru_cell_1_3/ReadVariableOp
gru_cell_1_3/unstackUnpack#gru_cell_1_3/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell_1_3/unstackз
"gru_cell_1_3/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource#^gru_cell_1_2/MatMul/ReadVariableOp*
_output_shapes

:`*
dtype02$
"gru_cell_1_3/MatMul/ReadVariableOpЌ
gru_cell_1_3/MatMulMatMuldense_4/BiasAdd:output:0*gru_cell_1_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMulЇ
gru_cell_1_3/BiasAddBiasAddgru_cell_1_3/MatMul:product:0gru_cell_1_3/unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAddj
gru_cell_1_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1_3/Const
gru_cell_1_3/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
gru_cell_1_3/split/split_dimр
gru_cell_1_3/splitSplit%gru_cell_1_3/split/split_dim:output:0gru_cell_1_3/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/splitп
$gru_cell_1_3/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource%^gru_cell_1_2/MatMul_1/ReadVariableOp*
_output_shapes

: `*
dtype02&
$gru_cell_1_3/MatMul_1/ReadVariableOpА
gru_cell_1_3/MatMul_1MatMulgru_cell_1_2/add_3:z:0,gru_cell_1_3/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/MatMul_1­
gru_cell_1_3/BiasAdd_1BiasAddgru_cell_1_3/MatMul_1:product:0gru_cell_1_3/unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
gru_cell_1_3/BiasAdd_1
gru_cell_1_3/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2
gru_cell_1_3/Const_1
gru_cell_1_3/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2 
gru_cell_1_3/split_1/split_dim
gru_cell_1_3/split_1SplitVgru_cell_1_3/BiasAdd_1:output:0gru_cell_1_3/Const_1:output:0'gru_cell_1_3/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
gru_cell_1_3/split_1
gru_cell_1_3/addAddV2gru_cell_1_3/split:output:0gru_cell_1_3/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add
gru_cell_1_3/SigmoidSigmoidgru_cell_1_3/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid
gru_cell_1_3/add_1AddV2gru_cell_1_3/split:output:1gru_cell_1_3/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_1
gru_cell_1_3/Sigmoid_1Sigmoidgru_cell_1_3/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Sigmoid_1
gru_cell_1_3/mulMulgru_cell_1_3/Sigmoid_1:y:0gru_cell_1_3/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul
gru_cell_1_3/add_2AddV2gru_cell_1_3/split:output:2gru_cell_1_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_2x
gru_cell_1_3/TanhTanhgru_cell_1_3/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/Tanh
gru_cell_1_3/mul_1Mulgru_cell_1_3/Sigmoid:y:0gru_cell_1_2/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_1m
gru_cell_1_3/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1_3/sub/x
gru_cell_1_3/subSubgru_cell_1_3/sub/x:output:0gru_cell_1_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/sub
gru_cell_1_3/mul_2Mulgru_cell_1_3/sub:z:0gru_cell_1_3/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/mul_2
gru_cell_1_3/add_3AddV2gru_cell_1_3/mul_1:z:0gru_cell_1_3/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
gru_cell_1_3/add_3Ѕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMulgru_cell_1_3/add_3:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/MatMulЄ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpЁ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_5/BiasAddє
tf_op_layer_packed/packedPackdense_2/BiasAdd:output:0dense_3/BiasAdd:output:0dense_4/BiasAdd:output:0dense_5/BiasAdd:output:0*
N*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2
tf_op_layer_packed/packedЁ
$tf_op_layer_transpose/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$tf_op_layer_transpose/transpose/permч
tf_op_layer_transpose/transpose	Transpose"tf_op_layer_packed/packed:output:0-tf_op_layer_transpose/transpose/perm:output:0*
T0*
_cloned(*+
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_transpose/transposeќ
IdentityIdentity#tf_op_layer_transpose/transpose:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp#^gru_cell_1_1/MatMul/ReadVariableOp%^gru_cell_1_1/MatMul_1/ReadVariableOp^gru_cell_1_1/ReadVariableOp#^gru_cell_1_2/MatMul/ReadVariableOp%^gru_cell_1_2/MatMul_1/ReadVariableOp^gru_cell_1_2/ReadVariableOp#^gru_cell_1_3/MatMul/ReadVariableOp%^gru_cell_1_3/MatMul_1/ReadVariableOp^gru_cell_1_3/ReadVariableOp^rnn/MatMul/ReadVariableOp^rnn/MatMul_1/ReadVariableOp^rnn/ReadVariableOp
^rnn/while*
T0*+
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:џџџџџџџџџ::::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2H
"gru_cell_1_1/MatMul/ReadVariableOp"gru_cell_1_1/MatMul/ReadVariableOp2L
$gru_cell_1_1/MatMul_1/ReadVariableOp$gru_cell_1_1/MatMul_1/ReadVariableOp2:
gru_cell_1_1/ReadVariableOpgru_cell_1_1/ReadVariableOp2H
"gru_cell_1_2/MatMul/ReadVariableOp"gru_cell_1_2/MatMul/ReadVariableOp2L
$gru_cell_1_2/MatMul_1/ReadVariableOp$gru_cell_1_2/MatMul_1/ReadVariableOp2:
gru_cell_1_2/ReadVariableOpgru_cell_1_2/ReadVariableOp2H
"gru_cell_1_3/MatMul/ReadVariableOp"gru_cell_1_3/MatMul/ReadVariableOp2L
$gru_cell_1_3/MatMul_1/ReadVariableOp$gru_cell_1_3/MatMul_1/ReadVariableOp2:
gru_cell_1_3/ReadVariableOpgru_cell_1_3/ReadVariableOp26
rnn/MatMul/ReadVariableOprnn/MatMul/ReadVariableOp2:
rnn/MatMul_1/ReadVariableOprnn/MatMul_1/ReadVariableOp2(
rnn/ReadVariableOprnn/ReadVariableOp2
	rnn/while	rnn/while:' #
!
_user_specified_name	input_1
ѓ
ш
while_cond_201017
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_201017___redundant_placeholder0.
*while_cond_201017___redundant_placeholder1.
*while_cond_201017___redundant_placeholder2.
*while_cond_201017___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
п
Ш
seq2seq_rnn_while_cond_196838"
seq2seq_rnn_while_loop_counter(
$seq2seq_rnn_while_maximum_iterations
placeholder
placeholder_1
placeholder_2$
 less_seq2seq_rnn_strided_slice_1:
6seq2seq_rnn_while_cond_196838___redundant_placeholder0:
6seq2seq_rnn_while_cond_196838___redundant_placeholder1:
6seq2seq_rnn_while_cond_196838___redundant_placeholder2:
6seq2seq_rnn_while_cond_196838___redundant_placeholder3
identity
d
LessLessplaceholder less_seq2seq_rnn_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::
ѓQ

$__inference_rnn_layer_call_fn_200631

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1ЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЂwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
TensorArrayV2/element_shapeВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2П
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeј
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ќ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    2
TensorArrayV2_1/element_shapeИ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЂ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
bodyR
while_body_200541*
condR
while_cond_200540*8
output_shapes'
%: : : : :џџџџџџџџџ : : : : : *
parallel_iterations 2
whileЕ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permЅ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
transpose_1В
IdentityIdentitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*+
_output_shapes
:џџџџџџџџџ 2

IdentityБ

Identity_1Identitywhile:output:4^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:џџџџџџџџџ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
ш
м
C__inference_dense_4_layer_call_and_return_conditional_losses_201636

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
О1
И
while_body_201177
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resourceЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpЂReadVariableOpЗ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   23
1TensorArrayV2Read/TensorListGetItem/element_shapeЕ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOp
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split/split_dimЌ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        џџџџ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2
split_1/split_dimг
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
add_3Е
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2О

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3Є

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :џџџџџџџџџ : : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
ѓ
ш
while_cond_200858
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_200858___redundant_placeholder0.
*while_cond_200858___redundant_placeholder1.
*while_cond_200858___redundant_placeholder2.
*while_cond_200858___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :џџџџџџџџџ : ::::"Џ-
saver_filename:0
Identity:0Identity_158"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Р
serving_defaultЌ
?
input_14
serving_default_input_1:0џџџџџџџџџM
tf_op_layer_transpose4
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:До
Ќe
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"Сa
_tf_keras_modelЇa{"class_name": "Model", "name": "seq2seq", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "seq2seq", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_1", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"shrink_axis_mask": {"i": "6"}, "begin_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, -1, 0], "2": [0, 0, 1], "3": [1, 1, 1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["strided_slice", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}}, "name": "rnn", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "GRUCell", "config": {"name": "gru_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_cell_1", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {"states": [["rnn", 0, 1]]}]], [["dense_2", 0, 0, {"states": [["gru_cell_1", 0, 1]]}]], [["dense_3", 0, 0, {"states": [["gru_cell_1", 1, 1]]}]], [["dense_4", 0, 0, {"states": [["gru_cell_1", 2, 1]]}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["gru_cell_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["gru_cell_1", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["gru_cell_1", 2, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["gru_cell_1", 3, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "packed", "trainable": true, "dtype": "float32", "node_def": {"name": "packed", "op": "Pack", "input": ["while/dense_2/Identity", "while/dense_3/Identity", "while/dense_4/Identity", "while/dense_5/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "axis": {"i": "0"}, "N": {"i": "4"}}}, "constants": {}}, "name": "tf_op_layer_packed", "inbound_nodes": [[["dense_2", 0, 0, {}], ["dense_3", 0, 0, {}], ["dense_4", 0, 0, {}], ["dense_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "transpose", "trainable": true, "dtype": "float32", "node_def": {"name": "transpose", "op": "Transpose", "input": ["packed", "transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0, 2]}}, "name": "tf_op_layer_transpose", "inbound_nodes": [[["tf_op_layer_packed", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf_op_layer_transpose", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "seq2seq", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_1", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"shrink_axis_mask": {"i": "6"}, "begin_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, -1, 0], "2": [0, 0, 1], "3": [1, 1, 1]}}, "name": "tf_op_layer_strided_slice", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["strided_slice", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}}, "name": "rnn", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "GRUCell", "config": {"name": "gru_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "name": "gru_cell_1", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {"states": [["rnn", 0, 1]]}]], [["dense_2", 0, 0, {"states": [["gru_cell_1", 0, 1]]}]], [["dense_3", 0, 0, {"states": [["gru_cell_1", 1, 1]]}]], [["dense_4", 0, 0, {"states": [["gru_cell_1", 2, 1]]}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["gru_cell_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["gru_cell_1", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["gru_cell_1", 2, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["gru_cell_1", 3, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "packed", "trainable": true, "dtype": "float32", "node_def": {"name": "packed", "op": "Pack", "input": ["while/dense_2/Identity", "while/dense_3/Identity", "while/dense_4/Identity", "while/dense_5/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "axis": {"i": "0"}, "N": {"i": "4"}}}, "constants": {}}, "name": "tf_op_layer_packed", "inbound_nodes": [[["dense_2", 0, 0, {}], ["dense_3", 0, 0, {}], ["dense_4", 0, 0, {}], ["dense_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "transpose", "trainable": true, "dtype": "float32", "node_def": {"name": "transpose", "op": "Transpose", "input": ["packed", "transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0, 2]}}, "name": "tf_op_layer_transpose", "inbound_nodes": [[["tf_op_layer_packed", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf_op_layer_transpose", 0, 0]]}}}
Ѕ"Ђ
_tf_keras_input_layer{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 16, 1], "config": {"batch_input_shape": [null, 16, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
з
	constants
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"З
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "strided_slice", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["input_1", "strided_slice/begin", "strided_slice/end", "strided_slice/strides"], "attr": {"shrink_axis_mask": {"i": "6"}, "begin_mask": {"i": "1"}, "ellipsis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, -1, 0], "2": [0, 0, 1], "3": [1, 1, 1]}}}
щ
	constants
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЏ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["strided_slice", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
Я

cell

state_spec
	variables
regularization_losses
trainable_variables
 	keras_api
__call__
+&call_and_return_all_conditional_losses"Є	
_tf_keras_layer	{"class_name": "RNN", "name": "rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": true, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 1], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
ь

!kernel
"recurrent_kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"Џ
_tf_keras_layer{"class_name": "GRUCell", "name": "gru_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
є

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
__call__
+&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
є

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
__call__
+&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
є

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
__call__
+&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
є

:kernel
;bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
__call__
+&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}

@	constants
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
__call__
+&call_and_return_all_conditional_losses"§
_tf_keras_layerу{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_packed", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "packed", "trainable": true, "dtype": "float32", "node_def": {"name": "packed", "op": "Pack", "input": ["while/dense_2/Identity", "while/dense_3/Identity", "while/dense_4/Identity", "while/dense_5/Identity"], "attr": {"T": {"type": "DT_FLOAT"}, "axis": {"i": "0"}, "N": {"i": "4"}}}, "constants": {}}}
ч
E	constants
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
__call__
+&call_and_return_all_conditional_losses"Ч
_tf_keras_layer­{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "transpose", "trainable": true, "dtype": "float32", "node_def": {"name": "transpose", "op": "Transpose", "input": ["packed", "transpose/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0, 2]}}}

J0
K1
L2
!3
"4
#5
(6
)7
.8
/9
410
511
:12
;13"
trackable_list_wrapper
 "
trackable_list_wrapper

J0
K1
L2
!3
"4
#5
(6
)7
.8
/9
410
511
:12
;13"
trackable_list_wrapper
Л
Mlayer_regularization_losses
Nnon_trainable_variables
Ometrics
	variables
regularization_losses

Players
trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Qlayer_regularization_losses
Rnon_trainable_variables
Smetrics
	variables
regularization_losses

Tlayers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
	variables
regularization_losses

Xlayers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш

Jkernel
Krecurrent_kernel
Lbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
__call__
+&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper

]layer_regularization_losses
^non_trainable_variables
_metrics
	variables
regularization_losses

`layers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'`2while/gru_cell_1/kernel
3:1 `2!while/gru_cell_1/recurrent_kernel
':%`2while/gru_cell_1/bias
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper

alayer_regularization_losses
bnon_trainable_variables
cmetrics
$	variables
%regularization_losses

dlayers
&trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$ 2while/dense_2/kernel
 :2while/dense_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper

elayer_regularization_losses
fnon_trainable_variables
gmetrics
*	variables
+regularization_losses

hlayers
,trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$ 2while/dense_3/kernel
 :2while/dense_3/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper

ilayer_regularization_losses
jnon_trainable_variables
kmetrics
0	variables
1regularization_losses

llayers
2trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$ 2while/dense_4/kernel
 :2while/dense_4/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper

mlayer_regularization_losses
nnon_trainable_variables
ometrics
6	variables
7regularization_losses

players
8trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$ 2while/dense_5/kernel
 :2while/dense_5/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper

qlayer_regularization_losses
rnon_trainable_variables
smetrics
<	variables
=regularization_losses

tlayers
>trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

ulayer_regularization_losses
vnon_trainable_variables
wmetrics
A	variables
Bregularization_losses

xlayers
Ctrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

ylayer_regularization_losses
znon_trainable_variables
{metrics
F	variables
Gregularization_losses

|layers
Htrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:`2
rnn/kernel
&:$ `2rnn/recurrent_kernel
:`2rnn/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper

}layer_regularization_losses
~non_trainable_variables
metrics
Y	variables
Zregularization_losses
layers
[trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
(__inference_seq2seq_layer_call_fn_199788
(__inference_seq2seq_layer_call_fn_200106Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
!__inference__wrapped_model_197082К
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ **Ђ'
%"
input_1џџџџџџџџџ
а2Э
C__inference_seq2seq_layer_call_and_return_conditional_losses_199469
C__inference_seq2seq_layer_call_and_return_conditional_losses_199151Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ф2с
:__inference_tf_op_layer_strided_slice_layer_call_fn_200142Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џ2ќ
U__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_200134Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
7__inference_tf_op_layer_ExpandDims_layer_call_fn_200154Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
R__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_200148Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
$__inference_rnn_layer_call_fn_201426
$__inference_rnn_layer_call_fn_201267
$__inference_rnn_layer_call_fn_200790
$__inference_rnn_layer_call_fn_200631ц
нВй
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
№2э
?__inference_rnn_layer_call_and_return_conditional_losses_200313
?__inference_rnn_layer_call_and_return_conditional_losses_200949
?__inference_rnn_layer_call_and_return_conditional_losses_200472
?__inference_rnn_layer_call_and_return_conditional_losses_201108ц
нВй
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
+__inference_gru_cell_1_layer_call_fn_201586
+__inference_gru_cell_1_layer_call_fn_201546О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201466
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201506О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
(__inference_dense_2_layer_call_fn_201606Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_201596Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_3_layer_call_fn_201626Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_3_layer_call_and_return_conditional_losses_201616Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_4_layer_call_fn_201646Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_4_layer_call_and_return_conditional_losses_201636Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_5_layer_call_fn_201666Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_5_layer_call_and_return_conditional_losses_201656Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н2к
3__inference_tf_op_layer_packed_layer_call_fn_201682Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_tf_op_layer_packed_layer_call_and_return_conditional_losses_201674Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
р2н
6__inference_tf_op_layer_transpose_layer_call_fn_201694Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћ2ј
Q__inference_tf_op_layer_transpose_layer_call_and_return_conditional_losses_201688Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_200126input_1
2
)__inference_gru_cell_layer_call_fn_201854
)__inference_gru_cell_layer_call_fn_201814О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
D__inference_gru_cell_layer_call_and_return_conditional_losses_201734
D__inference_gru_cell_layer_call_and_return_conditional_losses_201774О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 П
!__inference__wrapped_model_197082LJK#!"()./45:;4Ђ1
*Ђ'
%"
input_1џџџџџџџџџ
Њ "QЊN
L
tf_op_layer_transpose30
tf_op_layer_transposeџџџџџџџџџЃ
C__inference_dense_2_layer_call_and_return_conditional_losses_201596\()/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_2_layer_call_fn_201606O()/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЃ
C__inference_dense_3_layer_call_and_return_conditional_losses_201616\.//Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_3_layer_call_fn_201626O.//Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЃ
C__inference_dense_4_layer_call_and_return_conditional_losses_201636\45/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_4_layer_call_fn_201646O45/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЃ
C__inference_dense_5_layer_call_and_return_conditional_losses_201656\:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_5_layer_call_fn_201666O:;/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201466З#!"\ЂY
RЂO
 
inputsџџџџџџџџџ
'$
"
states/0џџџџџџџџџ 
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 
F__inference_gru_cell_1_layer_call_and_return_conditional_losses_201506З#!"\ЂY
RЂO
 
inputsџџџџџџџџџ
'$
"
states/0џџџџџџџџџ 
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 й
+__inference_gru_cell_1_layer_call_fn_201546Љ#!"\ЂY
RЂO
 
inputsџџџџџџџџџ
'$
"
states/0џџџџџџџџџ 
p
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ й
+__inference_gru_cell_1_layer_call_fn_201586Љ#!"\ЂY
RЂO
 
inputsџџџџџџџџџ
'$
"
states/0џџџџџџџџџ 
p 
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ 
D__inference_gru_cell_layer_call_and_return_conditional_losses_201734ЗLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ 
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 
D__inference_gru_cell_layer_call_and_return_conditional_losses_201774ЗLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ 
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ 
$!

0/1/0џџџџџџџџџ 
 з
)__inference_gru_cell_layer_call_fn_201814ЉLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ 
p
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ з
)__inference_gru_cell_layer_call_fn_201854ЉLJK\ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ 
p 
Њ "DЂA

0џџџџџџџџџ 
"

1/0џџџџџџџџџ п
?__inference_rnn_layer_call_and_return_conditional_losses_200313LJKCЂ@
9Ђ6
$!
inputsџџџџџџџџџ

 
p

 

 
Њ "OЂL
EB
!
0/0џџџџџџџџџ 

0/1џџџџџџџџџ 
 п
?__inference_rnn_layer_call_and_return_conditional_losses_200472LJKCЂ@
9Ђ6
$!
inputsџџџџџџџџџ

 
p 

 

 
Њ "OЂL
EB
!
0/0џџџџџџџџџ 

0/1џџџџџџџџџ 
 ј
?__inference_rnn_layer_call_and_return_conditional_losses_200949ДLJKSЂP
IЂF
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 

 
Њ "XЂU
NK
*'
0/0џџџџџџџџџџџџџџџџџџ 

0/1џџџџџџџџџ 
 ј
?__inference_rnn_layer_call_and_return_conditional_losses_201108ДLJKSЂP
IЂF
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 

 
Њ "XЂU
NK
*'
0/0џџџџџџџџџџџџџџџџџџ 

0/1џџџџџџџџџ 
 Ж
$__inference_rnn_layer_call_fn_200631LJKCЂ@
9Ђ6
$!
inputsџџџџџџџџџ

 
p

 

 
Њ "A>

0џџџџџџџџџ 

1џџџџџџџџџ Ж
$__inference_rnn_layer_call_fn_200790LJKCЂ@
9Ђ6
$!
inputsџџџџџџџџџ

 
p 

 

 
Њ "A>

0џџџџџџџџџ 

1џџџџџџџџџ Я
$__inference_rnn_layer_call_fn_201267ІLJKSЂP
IЂF
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 

 
Њ "JG
(%
0џџџџџџџџџџџџџџџџџџ 

1џџџџџџџџџ Я
$__inference_rnn_layer_call_fn_201426ІLJKSЂP
IЂF
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 

 
Њ "JG
(%
0џџџџџџџџџџџџџџџџџџ 

1џџџџџџџџџ Р
C__inference_seq2seq_layer_call_and_return_conditional_losses_199151yLJK#!"()./45:;<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Р
C__inference_seq2seq_layer_call_and_return_conditional_losses_199469yLJK#!"()./45:;<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 
(__inference_seq2seq_layer_call_fn_199788lLJK#!"()./45:;<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
(__inference_seq2seq_layer_call_fn_200106lLJK#!"()./45:;<Ђ9
2Ђ/
%"
input_1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџЭ
$__inference_signature_wrapper_200126ЄLJK#!"()./45:;?Ђ<
Ђ 
5Њ2
0
input_1%"
input_1џџџџџџџџџ"QЊN
L
tf_op_layer_transpose30
tf_op_layer_transposeџџџџџџџџџБ
R__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_200148[2Ђ/
(Ђ%
# 

inputs/0џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
7__inference_tf_op_layer_ExpandDims_layer_call_fn_200154N2Ђ/
(Ђ%
# 

inputs/0џџџџџџџџџ
Њ "џџџџџџџџџЈ
N__inference_tf_op_layer_packed_layer_call_and_return_conditional_losses_201674еЇЂЃ
Ђ

"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
3__inference_tf_op_layer_packed_layer_call_fn_201682ШЇЂЃ
Ђ

"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
"
inputs/3џџџџџџџџџ
Њ "џџџџџџџџџИ
U__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_200134_:Ђ7
0Ђ-
+(
&#
inputs/0џџџџџџџџџ
Њ "!Ђ

0џџџџџџџџџ
 
:__inference_tf_op_layer_strided_slice_layer_call_fn_200142R:Ђ7
0Ђ-
+(
&#
inputs/0џџџџџџџџџ
Њ "џџџџџџџџџМ
Q__inference_tf_op_layer_transpose_layer_call_and_return_conditional_losses_201688g:Ђ7
0Ђ-
+(
&#
inputs/0џџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
6__inference_tf_op_layer_transpose_layer_call_fn_201694Z:Ђ7
0Ђ-
+(
&#
inputs/0џџџџџџџџџ
Њ "џџџџџџџџџ