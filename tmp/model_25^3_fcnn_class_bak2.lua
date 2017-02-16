--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'
require 'cudnn'
require './util'
require 'nnx'

if not opt then
    opt = {multiGPU = true,batchSize=2}
end

print('===> load pre-trained model...');
collectgarbage();
modelPath = 'training/25^3_i9_r0.05/model_10.net'
--modelPath = 'training/25^3_fcnn_i12_r0.1_w150/model_11.net'

fcnnModel = torch.load(modelPath);

p,gp = fcnnModel:parameters();

for i=1,#p do
    p[i] = p[i]:double();
end

fcnnModel = nil;

collectgarbage();

print('===>loading model...');


-- building block
function ConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

function DeConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

function ConvBNReLUStride2(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

function DeConvBNReLUStride2(convNet,nInputPlane, nOutputPlane, adjz,adj)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1,adjz,adj,adj))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

function wrapGPU(model, inGPU, outGPU)
    if opt.multiGPU then
        if outGPU then
            return nn.GPU(model,inGPU, outGPU);
        else
            return nn.GPU(model,inGPU);
        end
    else
        return model;
    end
end


MaxPooling = nn.VolumetricMaxPooling
UnPooling = nn.VolumetricMaxUnpooling


convGpu1 = nn.Sequential();
ConvBNReLU(convGpu1,1,32):add(nn.Dropout(0.3));
convGpu1 = wrapGPU(convGpu1,1);
mp0 = MaxPooling(2,2,2,2,2,2);
mp0 = wrapGPU(mp0,4);


convGpu1_2 = nn.Sequential();
ConvBNReLU(convGpu1_2,32,64):add(nn.Dropout(0.4));
convGpu1_2 = wrapGPU(convGpu1_2,3);
mp1 = MaxPooling(2, 2, 2, 2, 2, 2);
mp1 = wrapGPU(mp1,4);


convGpu2 = nn.Sequential();
ConvBNReLU(convGpu2,64, 128):add(nn.Dropout(0.4))
ConvBNReLU(convGpu2,128, 128)
convGpu2 = wrapGPU(convGpu2,3)

mp2 = MaxPooling(2, 2, 2, 2, 2, 2);
mp2 = wrapGPU(mp2,4)

convGpu2_2 = nn.Sequential();
ConvBNReLU(convGpu2_2, 128, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256);
convGpu2_2 = wrapGPU(convGpu2_2,3)

mp3 = MaxPooling(1, 2, 2, 1, 2, 2);
mp3 = wrapGPU(mp3,4)

convGpu3 = nn.Sequential();
ConvBNReLU(convGpu3, 256, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512);
convGpu3 = wrapGPU(convGpu3,4);

mp4 = MaxPooling(1, 2, 2, 1, 2, 2);
mp4 = wrapGPU(mp4,4)


convGpu3_2 = nn.Sequential();
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512)
convGpu3_2 = wrapGPU(convGpu3_2,4)
--
mp5 = MaxPooling(1, 2, 2, 1, 2, 2);
mp5 = wrapGPU(mp5,4);



fcnn = nn.Sequential();
fcnn:add(convGpu1):add(mp0)
fcnn:add(convGpu1_2):add(mp1):add(convGpu2):add(mp2):add(convGpu2_2):add(mp3);
fcnn:add(convGpu3):add(mp4):add(convGpu3_2):add(mp5);

convGpu4 = nn.Sequential();
up1 = UnPooling(mp5.modules[1]);
convGpu4:add(up1);
cat1 = nn.ConcatTable();
cat1:add(convGpu4);
cat1:add(nn.Identity());
cat1 = wrapGPU(cat1,4)
fcnn:add(cat1);

deGroup1 = nn.Sequential();
DeConvBNReLU(deGroup1,512,512);
up2 = UnPooling(mp4.modules[1]);
deGroup1:add(up2);
cat2 = nn.ConcatTable();
cat2:add(deGroup1);
cat2:add(nn.Identity());
cat2 = wrapGPU(cat2,4);
convGpu4:add(cat2);

deGroup2 = nn.Sequential();
DeConvBNReLU(deGroup2,512,256);
up3 = UnPooling(mp3.modules[1]);
deGroup2:add(up3);
cat3 = nn.ConcatTable();
cat3:add(deGroup2);
cat3:add(nn.Identity());
cat3 = wrapGPU(cat3,4);
deGroup1:add(cat3);

deGroup3 = nn.Sequential();
DeConvBNReLU(deGroup3,256,128);
deGroup3:add(UnPooling(mp2.modules[1]));
cat4 = nn.ConcatTable();
cat4:add(deGroup3);
cat4:add(nn.Identity());
cat4 = wrapGPU(cat4,4);
deGroup2:add(cat4);

deGroup4 = nn.Sequential();
DeConvBNReLU(deGroup4,128,64);
deGroup4:add(UnPooling(mp1.modules[1]));
cat5 = nn.ConcatTable();
cat5:add(deGroup4);
cat5:add(nn.Identity());
cat5 = wrapGPU(cat5,4);
deGroup3:add(cat5);

deGroup5 = nn.Sequential();
DeConvBNReLUStride2(deGroup5,64,64,0,0);
cat6 = nn.ConcatTable();
cat6:add(deGroup5);
cat6:add(nn.Identity());
cat6 = wrapGPU(cat6,2);
deGroup4:add(cat6);

deGroup6 = nn.Sequential();
DeConvBNReLUStride2(deGroup6,64,2,1,1);
cat7 = nn.ConcatTable();
cat7:add(deGroup6);
cat7:add(nn.Identity());
cat7 = wrapGPU(cat7,4);
deGroup5:add(cat7);

finalGroup1 = nn.Sequential();
finalGroup1:add(nn.FlattenTable())

finalGroup1:add(nn.NarrowTable(4,5));

--{
--1 : CudaTensor - size: 2x128x6x63x63
--2 : CudaTensor - size: 2x256x3x31x31
--3 : CudaTensor - size: 2x512x3x15x15
--4 : CudaTensor - size: 2x512x3x7x7
--5 : CudaTensor - size: 2x512x3x3x3
--}



para1 = nn.ParallelTable();
fitTensor1 = nn.VolumetricReplicationPadding(0, 63-3, 0, 63-3, 0, 6-3);
fitTensor2 = nn.VolumetricReplicationPadding(0, 63-7, 0, 63-7, 0, 6-3);
fitTensor3 = nn.VolumetricReplicationPadding(0, 63-15, 0, 63-15, 0, 6-3);
fitTensor4 = nn.VolumetricReplicationPadding(0, 63-31, 0, 63-31, 0, 6-3);
fitTensor5 = nn.Identity();
para1:add(fitTensor5);
para1:add(fitTensor4);
para1:add(fitTensor3);
para1:add(fitTensor2);
para1:add(fitTensor1);

finalGroup1:add(para1);
finalGroup1 = wrapGPU(finalGroup1,3);

fcnn:add(finalGroup1);


fcnn:add(wrapGPU(nn.JoinTable(2),4,1));


finalGroup2 = nn.Sequential();
finalGroup2:add(nn.VolumetricConvolution(1920,512,1,1,1,1,1,1))
finalGroup2:add(nn.VolumetricBatchNormalization(512, 1e-3))
finalGroup2:add(nn.ReLU(true))
finalGroup2:add(nn.VolumetricConvolution(512,16,1,1,1,1,1,1))
finalGroup2:add(nn.VolumetricBatchNormalization(16, 1e-3))
finalGroup2:add(nn.ReLU(true))

--finalGroup2:add(nn.View(config.batch,64*2*15*15))
finalGroup2:add(nn.View(opt.batchSize,16*6*63*63))
finalGroup2:add(nn.Linear(16*6*63*63,512))
finalGroup2:add(nn.Linear(512,56))

fcnn:add(finalGroup2)

--1
--1920
--6
--63
--63
--result = finalGroup2:cuda():forward(torch.CudaTensor(2,1920,6,63,63));


fp,fgp = fcnn:parameters();

for i=1,#p do
    origParameters = p[i];
    fp[i]:copy(origParameters);
end

collectgarbage();
--collectgarbage();
fcnn = fcnn:cuda();
m = fcnn:forward(torch.CudaTensor(2,1,56,512,512));
--print(m:size());
----target = torch.CudaTensor(1,2,56,512,512):fill(0);
----fcnn:backward(m,target);
collectgarbage(); getMemStats()


return fcnn