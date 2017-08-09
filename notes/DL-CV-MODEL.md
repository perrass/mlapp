# DL-CV-MODEL

## AlexNet

5 conv + 3 fully + softmax

## VGG

VGG-16: 2conv/maxpooling->2conv/max->3conv/max->3conv/max->3conv/max->3 fully + softmax

VGG-19: 2conv/maxpooling->2conv/max->4conv/max->4conv/max->4conv/max->3 fully + softmax

## GoogLeNet

L1: conv/max->2conv/max-> 2inception/max -> inception -> average -> conv -> 1 fully + 1 linear + softmax

L2: conv/max->2conv/max-> 2inception/max -> 4inception -> average -> conv -> 1 full + 1 linear + softmax

L3: conv/max->2conv/max-> 2inception/max -> 6inception -> average -> 1 linear + softmax

**inception**: maxpooling -> 11 conv + 11 conv -> 33 conv + 11 conv -> 55 conv + maxpooling -> 11 conv (stack the results)

**11 conv**: 

**average pooling**: 

## ResNet