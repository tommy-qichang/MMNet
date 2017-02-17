--
-- Unet
-- Input:252x252x28
-- output: 68x68x12
--
require 'nn'
require 'torch'
require 'cunn'
require 'cudnn'
require './util'
--matio = require 'matio'
require 'optim'
require 'nngraph'


nngraph.setDebug(true)

cutorch.setKernelPeerToPeerAccess(true)
if not opt then
    opt = {multiGPU = true}
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


function conv_relu_layers(fromopt, toopt,gpuid) -- two convolution/relu layers fromopt[1]->toopt[1]->toopt[1] with optional batch normalization (toopt.bn) and dropout (toopt.dropout).
    local from, to = fromopt[1], toopt[1]
--    local zStride = toopt.zs
    local model = nn.Sequential()
    if gpuid and gpuid[1] then

        local submodel = nn.Sequential()
        if fromopt.down then
            submodel:add(wrapGPU(nn.VolumetricConvolution(from, fromopt.down, 1, 1, 1, 1, 1, 1, 0, 0, 0),gpuid[1]))
            from = fromopt.down
        end
        submodel:add(wrapGPU(nn.VolumetricConvolution(from, to, 3, 3, 3, 1, 1, 1, 1, 1, 1),gpuid[1])) -- from->to, pad 1, same size

        if toopt.bn then submodel:add(wrapGPU(nn.VolumetricBatchNormalization(to),gpuid[3])) end
        if toopt.leaky then submodel:add(wrapGPU(nn.LReLU(toopt.leaky, true),gpuid[3])) else model:add(nn.ReLU(true)) end
        if toopt.dropout then submodel:add(wrapGPU(nn.Dropout(toopt.dropout),gpuid[3])) end

        model:add(submodel);
    else
        if fromopt.down then
            model:add(nn.VolumetricConvolution(from, fromopt.down, 1, 1, 1, 1, 1, 1, 0, 0, 0))
            from = fromopt.down
        end
        model:add(nn.VolumetricConvolution(from, to, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- from->to, pad 1, same size
        if toopt.bn then model:add(nn.VolumetricBatchNormalization(to)) end
        if toopt.leaky then model:add(nn.LReLU(toopt.leaky, true)) else model:add(nn.ReLU(true)) end
        if toopt.dropout then model:add(nn.Dropout(toopt.dropout)) end
    end


    if gpuid and gpuid[2] then

        local submodel = nn.Sequential()
        submodel:add(wrapGPU(nn.VolumetricConvolution(to, to, 3, 3, 3, 1, 1, 1, 1, 1, 1),gpuid[2])) -- from->to, pad 1, same size
        if toopt.bn then submodel:add(wrapGPU(nn.VolumetricBatchNormalization(to),gpuid[3])) end
        if toopt.leaky then submodel:add(wrapGPU(nn.LReLU(toopt.leaky, true),gpuid[3])) else model:add(nn.ReLU(true)) end
        if toopt.dropout then submodel:add(wrapGPU(nn.Dropout(toopt.dropout),gpuid[3])) end
        model:add(submodel);

    else
        model:add(nn.VolumetricConvolution(to, to, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- from->to, pad 1, same size
        if toopt.bn then model:add(nn.VolumetricBatchNormalization(to)) end
        if toopt.leaky then model:add(nn.LReLU(toopt.leaky, true)) else model:add(nn.ReLU(true)) end
        if toopt.dropout then model:add(nn.Dropout(toopt.dropout)) end
    end
    --model:add(nn.VolumetricConvolution(to, to, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- to -> to, pad 1, same size

    return model
end

local function create_unet_layer(layers,before) -- recursively create unet
    if #layers==2 then
        return conv_relu_layers(layers[1],layers[2])( before )
    end
    local from, to, tochild = layers[1], layers[2], layers[3]
    to.bn = to.bn or layers.bn -- allow common option for batch normalization
    local zStride = to.zs

    table.remove(layers,1)

    local new_before = conv_relu_layers(from,to)(before)
    local crop_new_before = nil;
    if to.crop then
        crop_new_before = nn.VolumetricReplicationPadding(to.crop,to.crop,to.crop,to.crop,to.zcrop,to.zcrop)(new_before);
    else
        crop_new_before = new_before;
    end

    to.zp = to.zp or to.zs;


    local child_model = create_unet_layer(layers, nn.VolumetricMaxPooling(to.zp,2,2,to.zp,2,2)( new_before ) )
    local joined_model = nn.JoinTable(1,4)({ crop_new_before, nn.ReLU(true)( nn.VolumetricFullConvolution(2*to[1],to[1],to.zp,2,2,to.zp,2,2)(child_model) ) })
    if from==1 then from=to end -- top layer keeps features one more conv.
    return conv_relu_layers(tochild,to)(joined_model)
end

local function nn_init(model)
    local function init_kaiming(name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW * v.kH * v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    init_kaiming('nn.SpatialConvolution')
    init_kaiming('nn.SpatialFullConvolution')
    local function init_bias(name)
        for k,v in pairs(model:findModules(name)) do
            v.bias:zero()
        end
    end
    init_bias('SpatialBatchNormalization')
end


function create_unet(layers)
    local first = layers[1][1]
    table.insert(layers,1,{1}) -- num channels = 1 assumed
    local input = nn.Identity()()
    local model = nn.gModule({input}, { nn.VolumetricConvolution(first,2, 1,3,3, 1,1,1,0,1,1)( create_unet_layer(layers,input) ) })
    nn_init(model)
    return model
end

--model = create_unet( {{32},{64},{128},{256},{512},bn=true} ):cuda();

--model = create_unet( {{32,crop=-88,zcrop=-8,zs=1},{64,crop=-40,zcrop=-8,zs=1},{128,crop=-16,zcrop=-8,zs=1},{256,crop=-4,zcrop=-4,zs=3,zp=2
--},{512,zs=3,zp=2},bn=true} ):cuda();


--mp = nn.VolumetricMaxPooling;
--up = nn.VolumetricFullConvolution

function mp(kT, kW, kH, dT, dW, dH)
    local mxp = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH);
    return wrapGPU(mxp,2);
end

function up(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH)
    local fc = nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH);
    return wrapGPU(fc,3);
end

input = nn.Identity()()
C1 = conv_relu_layers({1},{32,bn=true,dropout=0.3},{4,1,1})(input)

MP1 = mp(2,2,2,2,2,2)(C1)

C2 = conv_relu_layers({32},{64,bn=true,dropout=0.4},{2,2,3})(MP1)

MP2 = mp(2,2,2,2,2,2)(C2)

C3 = conv_relu_layers({64},{128,bn=true,dropout=0.4},{2,2,3})(MP2)

MP3 = mp(2,2,2,2,2,2)(C3)

C4 = conv_relu_layers({128},{256,bn=true,dropout=0.4},{2,2,4})(MP3)

MP4 = mp(1,2,2,1,2,2)(C4)

C5 = conv_relu_layers({256},{512,bn=true,dropout=0.4},{2,2,4})(MP4)

MP5 = mp(1,2,2,1,2,2)(C5)

C6 = conv_relu_layers({512},{1024,bn=true,dropout=0.4},{2,2,4})(MP5)


UP5 = up(1024,512,1,2,2,1,2,2)(C6)

JT5 = nn.JoinTable(1,4)({C5,UP5})

U5 = conv_relu_layers({1024,down=512},{512,bn=true,dropout=0.4},{2,2,2})(JT5)

UP4 = nn.VolumetricReplicationPadding(0,1,0,1,0,0)(up(512,256,1,2,2,1,2,2)(U5))

JT4 = nn.JoinTable(1,4)({C4,UP4})

U4 = conv_relu_layers({512,down=256},{256,bn=true,dropout=0.4},{2,2,2})(JT4)

UP3 = up(256,128,2,2,2,2,2,2)(U4)

JT3 = nn.JoinTable(1,4)({C3,UP3})

U3 = conv_relu_layers({256,down=128},{128,bn=true,dropout=0.4},{2,2,2})(JT3)

UP2 = up(128,64,2,2,2,2,2,2)(U3)

JT2 = nn.JoinTable(1,4)({C2,UP2})

U2 = conv_relu_layers({128,down=64},{64,bn=true,dropout=0.4},{2,2,2})(JT2)

UP1 = up(64,32,2,2,2,2,2,2)(U2)

JT1 = nn.JoinTable(1,4)({C1,UP1})

U1 = conv_relu_layers({64,down=32},{32,bn=true,dropout=0.4},{3,4,4})(JT1)

model = nn.gModule({input},{wrapGPU(nn.VolumetricConvolution(32,2, 1,1,1, 1,1,1,0,0,0),1)(U1)});
--model = nn.gModule({input},{U1});

graph.dot(model.fg,'vnet','vnet');



model = model:cuda();
collectgarbage();
input = torch.CudaTensor(1,1,40,200,200);
result = model:forward(input);
collectgarbage();
target = torch.CudaTensor(1,2,40,200,200):fill(0);
model:backward(result,target);
print(result:size());
getMemStats()
collectgarbage();

--graph.dot(model.fg,'model3d','u-net3d');

--[[
-- model = model:cuda();
collectgarbage();
input = torch.CudaTensor(1,1,28,252,252);

target = torch.CudaTensor(1,2,12,68,68):fill(0);

result = model:forward(input);
model:backward(result,target);

print('intput size:',input:size(),' and result/target size:',result:size());
collectgarbage(); getMemStats()
--]]


return model;












