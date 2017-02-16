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

if not opt then
    opt = {multiGPU = true}
end

--modelPath = 'training/25^3_fcnn_i15_r0.05_w50/model_19.net'
--
--fcnnModel = torch.load(modelPath);
--
--p,gp = fcnnModel:parameters();
--
--for i=1,#p do
--    p[i] = p[i]:double();
--end

--fcnnModel = nil;

-- building block
local function ConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local function DeConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local function ConvBNReLUStride2(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local function DeConvBNReLUStride2(convNet,nInputPlane, nOutputPlane, adjz,adj)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1,adjz,adj,adj))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local function ConvBNReLUStep1(convNet,nInputPlane,nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local function wrapGPU(model, inGPU, outGPU)
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


local MaxPooling = nn.VolumetricMaxPooling
local UnPooling = nn.VolumetricMaxUnpooling

local convGpu1 = nn.Sequential();
ConvBNReLUStride2(convGpu1,1,64):add(nn.Dropout(0.3));
local cat1 = nn.ConcatTable();
cat1:add(convGpu1);
cat1:add(nn.Identity());
cat1 = wrapGPU(cat1,1);

local convGpu1_2 = nn.Sequential();
ConvBNReLUStride2(convGpu1_2,64,64);
local mp1 = MaxPooling(2, 2, 2, 2, 2, 2);
convGpu1_2:add(mp1);
local cat1_2 = nn.ConcatTable();
cat1_2:add(convGpu1_2);
cat1_2:add(nn.Identity());
cat1_2 = wrapGPU(cat1_2,3);
convGpu1:add(cat1_2)




local convGpu2 = nn.Sequential();
ConvBNReLU(convGpu2,64, 128):add(nn.Dropout(0.4))
ConvBNReLU(convGpu2,128, 128)
local mp2 = MaxPooling(2, 2, 2, 2, 2, 2);
convGpu2:add(mp2);
local cat2 = nn.ConcatTable();
cat2:add(convGpu2);
cat2:add(nn.Identity());
cat2 = wrapGPU(cat2,4)
convGpu1_2:add(cat2);


local convGpu2_2 = nn.Sequential();
ConvBNReLU(convGpu2_2, 128, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256);
local mp3 = MaxPooling(1, 2, 2, 1, 2, 2);
convGpu2_2:add(mp3);
local cat2_2 = nn.ConcatTable();
cat2_2:add(convGpu2_2);
cat2_2:add(nn.Identity());
cat2_2 = wrapGPU(cat2_2,4)
convGpu2:add(cat2_2)

local convGpu3 = nn.Sequential();
ConvBNReLU(convGpu3, 256, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512);
local mp4 = MaxPooling(1, 2, 2, 1, 2, 2);
convGpu3:add(mp4);
local cat3 = nn.ConcatTable();
cat3:add(convGpu3);
cat3:add(nn.Identity());
cat3 = wrapGPU(cat3,4);
convGpu2_2:add(cat3);


local convGpu3_2 = nn.Sequential();
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512)
local mp5 = MaxPooling(1, 2, 2, 1, 2, 2);
convGpu3_2:add(mp5);
local cat3_2 = nn.ConcatTable();
cat3_2:add(convGpu3_2);
cat3_2:add(nn.Identity());
cat3_2 = wrapGPU(cat3_2,4)
convGpu3:add(cat3_2);
--

fcnn = nn.Sequential();
fcnn:add(cat1);


--1x512x3x3x3
--1x512x3x7x7
--1x256x3x15x15
--1x128x2x31x31
--1x64x6x63x63
--1x64x27x255x255
--1x1x56x512x512




local deConv1 = nn.Sequential();
deConv1:add(UnPooling(mp5));
DeConvBNReLU(deConv1,512,512);
convGpu3_2:add(deConv1);

--1x512x3x7x7
--1x512x3x7x7

local join1 = nn.JoinTable(2,5);
local deConv2 = nn.Sequential();
deConv2:add(UnPooling(mp4));
DeConvBNReLU(deConv2,1024,256);
convGpu3:add(join1);
convGpu3:add(deConv2);
--1x256x3x15x15  =x2

local join2 = nn.JoinTable(2,5);
local deConv3 = nn.Sequential();
deConv3:add(UnPooling(mp3));
DeConvBNReLU(deConv3,512,128);
convGpu2_2:add(join2);
convGpu2_2:add(deConv3);
--1x128x3x31x31 =x2

local join3 = nn.JoinTable(2,5);
local deConv4 = nn.Sequential();
deConv4:add(UnPooling(mp2));
DeConvBNReLU(deConv4,256,64);
convGpu2:add(join3);
convGpu2:add(deConv4);
--1x64x6x63x63 =x2

local join4 = nn.JoinTable(2,5);
local deConv5 = nn.Sequential();
deConv5:add(UnPooling(mp1));
DeConvBNReLU(deConv5,128,64);
convGpu1_2:add(join4);
convGpu1_2:add(deConv5);

local convGpu5 = nn.Sequential();
DeConvBNReLUStride2(convGpu5,64,64,0,0);
convGpu5 =wrapGPU(convGpu5,2)
convGpu1_2:add(convGpu5);
--1x64x27x255x255 =x2

local join5 = nn.JoinTable(2,5);
local convGpu6 = nn.Sequential();
DeConvBNReLUStride2(convGpu6,128,2,1,1);
convGpu6 =wrapGPU(convGpu6,4)
convGpu1:add(join5);
convGpu1:add(convGpu6);
--1x2x56x512x512 + 1x1x56x512x512

local join6 = nn.JoinTable(2,5);
fcnn:add(join6);

local convGPU6 = nn.Sequential();
ConvBNReLUStep1(convGPU6, 3, 2);
convGPU6 = wrapGPU(convGPU6,4,1);
fcnn:add(convGPU6);





-- load model parameters from already trained model for the first cnn part
--modelPath = 'training/25^3_i9_r0.05/model_10.net'
modelPath = 'training/25^3_i9_r0.05/model_2.net'
--modelPath = 'training/25^3_fcnn_i13_r0.1_w50/model_43.net'
--
vggmodel = torch.load(modelPath);
vggmodel:remove(52);
vggmodel:remove(51);
p,gp = vggmodel:parameters();
--
--
fp,fgp = fcnn:parameters();

for i=1,#p do
    origParameters = p[i];
    fp[i]:copy(origParameters);
end



--model = nn.Sequential();
--
--model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));

collectgarbage();
fcnn = fcnn:cuda();
--model:add(fcnn);
print('could not run...');
result = fcnn:forward(torch.CudaTensor(1,1,56,512,512));
print('actually could');
--result:getDevice();
--result:size();


collectgarbage();
fcnn = fcnn:cuda();
require('mobdebug').start(nill,8222)
print('2ed');
m = fcnn:forward(torch.CudaTensor(1,1,56,512,512));
print('2ed end');
target = torch.CudaTensor(1,2,56,512,512):fill(0);
fcnn:backward(m,target);
collectgarbage(); getMemStats()


return fcnn