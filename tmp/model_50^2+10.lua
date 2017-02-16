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

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
    vgg:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    vgg:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.ReLU(true))
    return vgg
end

local MaxPooling = nn.VolumetricMaxPooling

--input: Bx1x10x50x50
ConvBNReLU(1,64):add(nn.Dropout(0.3));
ConvBNReLU(64,64);
-- Bx64x10x50x50
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2):ceil());

-- Bx64x5x25x25
ConvBNReLU(64, 128):add(nn.Dropout(0.4))
ConvBNReLU(128, 128)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2):ceil())

-- Bx128x3x13x13
ConvBNReLU(128, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2):ceil())

-- Bx256x2x7x7
ConvBNReLU(256, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2):ceil())

-- Bx512x1x4x4
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(1, 2, 2, 1, 2, 2):ceil())
-- Bx512x1x2x2


ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(1, 2, 2, 1, 2, 2):ceil())
-- Bx512x1x1x1
vgg:add(nn.View(512))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 2))
vgg:add(classifier)


-- initialization from MSR
--local function MSRinit(net)
--    local function init(name)
--        for k, v in pairs(net:findModules(name)) do
--            local n = v.kW * v.kH * v.nOutputPlane
--            v.weight:normal(0, math.sqrt(2 / n))
--            v.bias:zero()
--        end
--    end
--
--    -- have to do for both backends
--    init 'nn.SpatialConvolution'
--end
--
--MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x2 tensor
--print(vgg:cuda():forward(torch.CudaTensor(16,1,10,50,50)))

return vgg