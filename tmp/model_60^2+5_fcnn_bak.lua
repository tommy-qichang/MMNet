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
vgg:add(ConvBNReLU(1,64):add(nn.Dropout(0.3)));

vgg:add(ConvBNReLU(64,64));
-- Bx64x5x60x60
mp1 = MaxPooling(2, 2, 2, 2, 2, 2);
vgg:add(mp1);

-- Bx64x2x30x30
vgg:add(ConvBNReLU(64, 128):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(128, 128))
mp2 = MaxPooling(1, 2, 2, 1, 2, 2);
vgg:add(mp2)

-- Bx128x1x15x15
vgg:add(ConvBNReLU(128, 256):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(256, 256):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(256, 256):add(nn.Dropout(0.4)))

mp3 = MaxPooling(1, 2, 2, 1, 2, 2);
vgg:add(mp3)

-- Bx256x1x7x7
vgg:add(ConvBNReLU(256, 512):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(512, 512):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(512, 512):add(nn.Dropout(0.4)))
mp4 = MaxPooling(1, 2, 2, 1, 2, 2);
vgg:add(mp4)

-- Bx512x1x3x3
vgg:add(ConvBNReLU(512, 512):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(512, 512):add(nn.Dropout(0.4)))
vgg:add(ConvBNReLU(512, 512))

decode:add(DeConvBNReLU(512,512));
decode:add(UnPooling(mp4));

decode:add(DeConvBNReLU(512,256));
decode:add(UnPooling(mp3));

decode:add(DeConvBNReLU(256,128));
decode:add(UnPooling(mp2));

decode:add(DeConvBNReLU(128,64));
decode:add(UnPooling(mp1));

decode:add(DeConvBNReLU(64,2));

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
print(fcnn:cuda():forward(torch.CudaTensor(1,1,2,256,256)):size())
--print(vgg:cuda():forward(torch.CudaTensor(1,1,16,256,256)):size())

return fcnn