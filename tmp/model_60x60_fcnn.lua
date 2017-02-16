--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'

vgg = nn.Sequential();
local decode = nn.Sequential();

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.ReLU(true))
    return vgg
end

local function DeConvBNReLU(nInputPlane, nOutputPlane)
    decode:add(nn.SpatialFullConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    decode:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    decode:add(nn.ReLU(true))
    return decode
end

local MaxPooling = nn.SpatialMaxPooling;
local UnPooling = nn.SpatialMaxUnpooling;

--input: Bx1x25x25
ConvBNReLU(1,64):add(nn.Dropout(0.3));
ConvBNReLU(64,64);
-- Bx64x25x25
mxp1 = MaxPooling(2, 2, 2, 2);
vgg:add(mxp1);

-- Bx64x13x13
ConvBNReLU(64, 128):add(nn.Dropout(0.4))
ConvBNReLU(128, 128)
mxp2 = MaxPooling(2, 2, 2, 2);
vgg:add(mxp2);

-- Bx128x7x7
ConvBNReLU(128, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
mxp3 = MaxPooling(2, 2, 2, 2);
vgg:add(mxp3)

-- Bx256x4x4
ConvBNReLU(256, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
mxp4 = MaxPooling(2, 2, 2, 2);
vgg:add(mxp4)

-- Bx512x2x2
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
-- Bx512x1x1

--vgg:add(nn.View(512*2*2))


DeConvBNReLU(512,512);
decode:add(UnPooling(mxp4));

DeConvBNReLU(512,256);
decode:add(UnPooling(mxp3));

DeConvBNReLU(256,128);
decode:add(UnPooling(mxp2));

DeConvBNReLU(128,64);
decode:add(UnPooling(mxp1));

DeConvBNReLU(64,2);

fcnn = nn.Sequential();
fcnn:add(vgg);
fcnn:add(decode);


--require('mobdebug').start(nill,8222)
print(fcnn:cuda():forward(torch.CudaTensor(3,1,256,256)):size())


return fcnn