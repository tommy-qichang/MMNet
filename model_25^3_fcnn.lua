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

--local modelPath = 'training/25^3_fcnn_i15_r0.05_w50/model_19.net'
--
--local fcnnModel = torch.load(modelPath);
--
--local p,gp = fcnnModel:parameters();
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


local MaxPooling = nn.VolumetricMaxPooling
local UnPooling = nn.VolumetricMaxUnpooling


local convGpu1 = nn.Sequential();
ConvBNReLU(convGpu1,1,32):add(nn.Dropout(0.3));

local convGpu1_2 = nn.Sequential();
ConvBNReLU(convGpu1_2,32,64):add(nn.Dropout(0.3));

local mp1 = MaxPooling(2, 2, 2, 2, 2, 2);

local convGpu2 = nn.Sequential();
ConvBNReLU(convGpu2,64, 128):add(nn.Dropout(0.4))
ConvBNReLU(convGpu2,128, 128)

local mp2 = MaxPooling(2, 2, 2, 2, 2, 2);

local convGpu2_2 = nn.Sequential();
ConvBNReLU(convGpu2_2, 128, 256):add(nn.Dropout(0.4));
--ConvBNReLU(convGpu2_2, 256, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256);

local mp3 = MaxPooling(1, 2, 2, 1, 2, 2);

--[[local convGpu3 = nn.Sequential();
ConvBNReLU(convGpu3, 256, 512):add(nn.Dropout(0.4));
--ConvBNReLU(convGpu3, 512, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512);

local mp4 = MaxPooling(1, 2, 2, 1, 2, 2);

]]
--[[local convGpu3_2 = nn.Sequential();
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
--ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512)]]
--
--local mp5 = MaxPooling(1, 2, 2, 1, 2, 2);
--mp5 = wrapGPU(mp5,4);

local convGpu4 = nn.Sequential();
--convGpu4:add(UnPooling(mp5.modules[1]));
--[[DeConvBNReLU(convGpu4,512,512);
convGpu4:add(UnPooling(mp4));
DeConvBNReLU(convGpu4,512,256);]]--
convGpu4:add(UnPooling(mp3));
DeConvBNReLU(convGpu4,256,128);
convGpu4:add(UnPooling(mp2));
DeConvBNReLU(convGpu4,128,64);
convGpu4:add(UnPooling(mp1));
--
local convGpu5 = nn.Sequential();

DeConvBNReLU(convGpu5,64,32,0,0);

local convGpu6 = nn.Sequential();
DeConvBNReLU(convGpu6,32,2,1,1);


fcnn = nn.Sequential();
fcnn:add(convGpu1):add(convGpu1_2):add(mp1):add(convGpu2):add(mp2):add(convGpu2_2):add(mp3);
fcnn:add(convGpu4):add(convGpu5):add(convGpu6);



------
--target = torch.CudaTensor(1,2,56,512,512):fill(0);
--fcnn:backward(m,target);
--collectgarbage(); getMemStats()


-- load model parameters from already trained model for the first cnn part
--local modelPath = 'training/25^3_i9_r0.05/model_10.net'
--local modelPath = 'training/25^3_i9_r0.05/model_2.net'
--local modelPath = 'training/25^3_fcnn_i13_r0.1_w50/model_43.net'
--
--local vggmodel = torch.load(modelPath);
--vggmodel:remove(52);
--vggmodel:remove(51);
--local p,gp = vggmodel:parameters();
--
--
--local fp,fgp = fcnn:parameters();

--for i=1,#p do
 --   local origParameters = p[i];
--    fp[i]:copy(origParameters);
--end




collectgarbage();
fcnn = fcnn:cuda();
m = fcnn:forward(torch.CudaTensor(1,1,20,200,200));
target = torch.CudaTensor(1,2,20,200,200):fill(0);
fcnn:backward(m,target);
collectgarbage(); getMemStats()


return fcnn