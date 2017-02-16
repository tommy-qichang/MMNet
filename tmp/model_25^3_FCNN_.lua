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
require 'nngraph'

function ConvBNReLUGraph(nInputPlane, nOutputPlane,input,dropout)
    if not input then
        input = nn.Identity()()
    end
    local vConv = nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1)(input);
    vConv:annotate{name='vConv'};
    local vBN = nn.VolumetricBatchNormalization(nOutputPlane, 1e-3)(vConv);
    local reLU = nn.ReLU(true)(vBN);
    local result = reLU;
    if dropout then
        result = nn.Dropout(0.3)(reLU);
    end
    return input,result;
end

function DeConvBNReLUGraph(nInputPlane, nOutputPlane,input)
    local deConv = nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1)(input);
    local deBN = nn.VolumetricBatchNormalization(nOutputPlane, 1e-3)(deConv);
    local reLU = nn.ReLU(true)(deBN);
    return input,reLU;
end


function MaxPoolingBranchGraph(input)
    local concat1 = nn.ConcatTable():add(nn.Identity()):add(nn.Identity())(input);
    local split1,split2 = concat1:split(2);
    local mp = nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2);
    local nMp = mp(split1);
    return mp,nMp,split2;
end

function MaxUnpoolingMergeGraph(mp,input1,input2)
    local mup1 = nn.VolumetricMaxUnpooling(mp)(input1);
    local jt1 = nn.JoinTable(1,4)({mup1,input2});
    return jt1;
end

--transferTensor = nn.Copy('torch.FloatTensor', 'torch.CudaTensor')();

local input,conv1 = ConvBNReLUGraph(1,64,nil,true);
local _,conv2 = ConvBNReLUGraph(64,64,conv1,false);
-- Bx64x25x25x25
local mp1,mp1_1,mp1_2 = MaxPoolingBranchGraph(conv2);
-- Bx64x12x12x12

local _,conv3 = ConvBNReLUGraph(64,128,mp1_1,true);
local _,conv4 = ConvBNReLUGraph(128,128,conv3,false);
local mp2,mp2_1,mp2_2 = MaxPoolingBranchGraph(conv4);
-- Bx128x6x6x6

local _,conv5 = ConvBNReLUGraph(128,256,mp2_1,true);
local _,conv6 = ConvBNReLUGraph(256,256,conv5,true);
local _,conv7 = ConvBNReLUGraph(256,256,conv6,false);
local mp3,mp3_1,mp3_2 = MaxPoolingBranchGraph(conv7);
-- Bx256x3x3x3

local _,conv8 = ConvBNReLUGraph(256,512,mp3_1,true);
local _,conv9 = ConvBNReLUGraph(512,512,conv8,true);
local _,conv10 = ConvBNReLUGraph(512,512,conv9,false);
local mp4,mp4_1,mp4_2 = MaxPoolingBranchGraph(conv10);
-- Bx512x1x1x1

-- Bx512x1x1x1
local _,deconv1 = DeConvBNReLUGraph(512,512,mp4_1);

local mup4 = MaxUnpoolingMergeGraph(mp4,deconv1,mp4_2);

-- Bx512x3x3x3 + Bx512x3x3x3 = Bx1024x3x3x3

local _,deconv2 = DeConvBNReLUGraph(1024,512,mup4)

-- Bx512x3x3x3

local mup3 = MaxUnpoolingMergeGraph(mp3,deconv2,mp3_2);

-- Bx512x6x6x6 + Bx256x6x6x6 = Bx768x6x6x6

local _,deconv3 = DeConvBNReLUGraph(768,256,mup3);

-- Bx256x6x6x6

local mup2 = MaxUnpoolingMergeGraph(mp2,deconv3,mp2_2);

-- Bx256x12x12x12 + Bx128x12x12x12 = Bx384x12x12x12

local _,deconv4 = DeConvBNReLUGraph(384,64,mup2);

-- Bx64x12x12x12

local mup1 = MaxUnpoolingMergeGraph(mp1,deconv4,mp1_2);

-- Bx64x25x25x25 + Bx64x25x25x25 = Bx128x25x25x25

local _,deconv5 = DeConvBNReLUGraph(128,64,mup1);

local _,deconv6 = DeConvBNReLUGraph(64,2,deconv5);

gmod = nn.gModule({input}, {deconv6});
--gmod = nn.gModule({input},{deconv1,mp4_2,mp3_2,mp2_2,mp1_2});
gmod = gmod:cuda();

result = gmod:forward(torch.zeros(1,1,26,128,128):cuda());
print(result:size());


return gmod;

--result = gmod:forward(torch.zeros(5,1,25,25,25):cuda());

