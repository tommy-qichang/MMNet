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

vgg = nn.Sequential();
decode = nn.Sequential();

-- building block
function ConvBNReLU(nInputPlane, nOutputPlane)
    convNet = nn.Sequential();
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

function DeConvBNReLU(nInputPlane, nOutputPlane)
    convNet = nn.Sequential()
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end


MaxPooling = nn.VolumetricMaxPooling
UnPooling = nn.VolumetricMaxUnpooling


--input: Bx1x5x60x60
vgg:add(nn.GPU(ConvBNReLU(1,64):add(nn.Dropout(0.3)),1,2));

--vgg:add(nn.GPU(nn.Sequential():add(nn.Identity()):add(nn.Identity())),2,3);


vgg:add(nn.GPU(ConvBNReLU(64,64),2,3));
-- Bx64x5x60x60
mp1 = MaxPooling(2, 2, 2, 2, 2, 2);
vgg:add(nn.GPU(mp1,3,4));

-- Bx64x2x30x30
vgg:add(nn.GPU(ConvBNReLU(64, 128):add(nn.Dropout(0.4)),4,1))
vgg:add(nn.GPU(ConvBNReLU(128, 128),1,2))
mp2 = MaxPooling(2, 2, 2, 2, 2, 2);
vgg:add(nn.GPU(mp2,2,3))

-- Bx128x1x15x15
vgg:add(nn.GPU(ConvBNReLU(128, 256):add(nn.Dropout(0.4)),3,2))
--vgg:add(nn.GPU(ConvBNReLU(256, 256):add(nn.Dropout(0.4)),4,1))
--vgg:add(nn.GPU(ConvBNReLU(256, 256):add(nn.Dropout(0.4)),1,2))

mp3 = MaxPooling(1, 2, 2, 1, 2, 2);
vgg:add(nn.GPU(mp3,2,3))

-- Bx256x1x7x7
vgg:add(nn.GPU(ConvBNReLU(256, 512):add(nn.Dropout(0.4)),3,2))
--vgg:add(nn.GPU(ConvBNReLU(512, 512):add(nn.Dropout(0.4)),4,1))
--vgg:add(nn.GPU(ConvBNReLU(512, 512):add(nn.Dropout(0.4)),1,2))
mp4 = MaxPooling(1, 2, 2, 1, 2, 2);
vgg:add(nn.GPU(mp4,2,3))

-- Bx512x1x3x3
vgg:add(nn.GPU(ConvBNReLU(512, 512):add(nn.Dropout(0.4)),3,4))
--vgg:add(nn.GPU(ConvBNReLU(512, 512):add(nn.Dropout(0.4)),4,1))
--vgg:add(nn.GPU(ConvBNReLU(512, 512),1,2))


decode:add(nn.GPU(DeConvBNReLU(512,512),2,2));
decode:add(nn.GPU(UnPooling(mp4),2,3));

decode:add(nn.GPU(DeConvBNReLU(512,256),3,2));
decode:add(nn.GPU(UnPooling(mp3),2,4));

decode:add(nn.GPU(DeConvBNReLU(256,128),4,2));
decode:add(nn.GPU(UnPooling(mp2),2,1));

decode:add(nn.GPU(DeConvBNReLU(128,64),1,3));
decode:add(nn.GPU(UnPooling(mp1),3,4));

decode:add(nn.GPU(DeConvBNReLU(64,2),4,1));

fcnn = nn.Sequential();
fcnn:add(vgg);
fcnn:add(decode);



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
print(fcnn:cuda():forward(torch.CudaTensor(1,1,20,256,256)):size())
--print(vgg:cuda():forward(torch.CudaTensor(1,1,16,256,256)):size())

return fcnn