--
--CUDA_VISIBLE_DEVICES=2 th -i voxel_predict_gen.lua  --backend=cudnn --save=analysis/25^3_i1_r0.01b0.5_val5 --model=model_25^3 --modelPath=training/model_10.net --batchSize=32 --learningRate=0.05 --balanceWeight=0.5 --valId=1
-- User: changqi
-- Date: 3/14/16
-- Time: 12:25 PM
-- To change this template use File | Settings | File Templates.
require 'nn';
require 'optim'
require 'cunn'
require 'image';
require 'cudnn'
require 'xlua';
local hdf5 = require 'hdf5'
dofile './provider_25^3.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "analysis/25^3")      subdirectory to save logs
   -b,--batchSize             (default 45)          batch size
   -r,--learningRate          (default 0.05)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default train_25^3)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   -i,--log_interval          (default 5)           show log interval
   --modelPath                (default training/model.net) exist model
   --balanceWeight              (default 1)     criterion balance weight
   --valId                    (default 1)    cross validation index
]]

print(opt)
------------------------------------ loading data----------------------------------------
print(c.blue '==>' .. ' loading data')
--provider = providerFactory(opt.valId);

local provider = Provider(opt.valId);
local saved = provider:load();

--provider = torch.load('provider.t7')
provider.dataset.trainData.data = provider.dataset.trainData.data:float()
provider.dataset.testData.data = provider.dataset.testData.data:float()


do -- data augmentation module
local BatchFlip, parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:__init()
    parent.__init(self)
    self.train = true
end

function BatchFlip:updateOutput(input)
    if self.train then
        local bs = input:size(1)
        local flip_mask = torch.randperm(bs):le(bs / 2)
        for i = 1, input:size(1) do
            if flip_mask[i] == 1 then
                input[i] =image.flip(input[i],4);
            end
        end
    end
    self.output:float():set(input);
    return self.output
end
end

------------------------------------ configuring----------------------------------------

print(c.red '==>' .. 'configuring model')
local modelPath = opt.modelPath;

local model = nn.Sequential();
model:add(nn.BatchFlip():float())

model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));

model:add(torch.load(modelPath));
print('==> load exist model:' .. modelPath);


model:get(1).updateGradInput = function(input) return end

----------------------------------- load exist model -------------------------------------





if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model:get(3), cudnn)
end

print(model);


parameters, gradParameters = model:getParameters()

------------------------------------ save log----------------------------------------
print('Will save at ' , opt.save)
paths.mkdir(opt.save)
--[[testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames { 'train 1st acc ', 'train 2ed acc' , 'test 1st acc', 'test 2ed acc','learning rate', 'costs'}
testLogger.showPlot = false]]

------------------------------------ set criterion---------------------------------------
--print(c.blue '==>' .. ' setting criterion')
--criterion = nn.CrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()

confusion = optim.ConfusionMatrix(2);
------------------------------------ optimizer config-------------------------------------
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}

evaln = 1;
cost = {}

function train()
    model:training();
    epoch = epoch or 1;

    -- drop learning rate every "epoch_step" epochs  ?
    --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate / 2 end

    -- update negative set every 6 epochs.
    if epoch % 6 == 0 then
        collectgarbage();
        print('...update data provider for negative data set...');
        provider:update();
        provider:normalize();
    end

    print(c.blue '==>' .. " online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


    local targets = torch.CudaTensor(opt.batchSize);
    -- random index and split all index into batches.
    local indices = torch.randperm(provider.dataset.trainData.data:size(1)):long():split(opt.batchSize);
    indices[#indices] = nil;


    local tic = torch.tic();
    local localCost = torch.zeros(#indices);
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)
        local innerTic = torch.tic();
        local inputs = provider.dataset.trainData.data:index(1, v);
        targets:copy(provider.dataset.trainData.labels:index(1, v));

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end

            gradParameters:zero();
            local outputs = model:forward(inputs:float())
            local f = criterion:forward(outputs, targets)

            local df = criterion:backward(outputs, targets)
            model:backward(inputs, df)


            confusion:batchAdd(outputs, targets);
            evaln = evaln+1;
            if evaln%800 == 0 then
                optimState.learningRate = optimState.learningRate*0.97;
            end
            localCost[t] = f;

            return f, gradParameters;
        end

        local x, fx = optim.sgd(feval, parameters, optimState);


        local innerToc = torch.toc(innerTic);
        local function printInfo()
            local tmpl = '---------%d/%d (epoch %.3f), ' ..
                    'train_loss = %6.8f, grad/param norm = %6.4e, ' ..
                    'speed = %5.1f/s, %5.3fs/iter -----------'
            print(string.format(tmpl,
                t, #indices, epoch,
                fx[1], gradParameters:norm() / parameters:norm(),
                opt.batchSize / innerToc, innerToc))
        end

        if t % opt.log_interval == 0 then
            printInfo();
        end
    end

    local avgCost = localCost:sum()/localCost:nElement();
    table.insert(cost,avgCost);
    --plotCost(opt.batchSize);
    print(c.blue '==>' .. 'Avg Cost for this round:'..avgCost);

    confusion:updateValids();
    print(c.red('Train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s, learning rate: %.6f'):format(confusion.totalValid * 100, torch.toc(tic),optimState.learningRate));
    print(confusion);

    train_acc = confusion.totalValid * 100

    firstClassAcc = confusion.valids[1];
    secondClassAcc = confusion.valids[2];
    confusion:zero()
    epoch = epoch + 1;
end




function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    local modelId = string.sub(opt.modelPath,-12,-5):gsub('/','');
    print(c.blue '==>' .. " testing and output... with modelID:"..modelId);

    local bs = opt.batchSize
    len = provider.dataset.testData.data:size(1);
    local allOutputs = torch.Tensor(len,2);
    for i = 1, len, bs do
        xlua.progress(i, len)
        if (i + bs) > len then idxEnd = len - i; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.dataset.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
        local outputs = model:forward(provider.dataset.testData.data:narrow(1, i, idxEnd or bs))
        allOutputs:narrow(1, i, idxEnd or bs):copy(outputs);
        confusion:batchAdd(outputs, provider.dataset.testData.labels:narrow(1, i, idxEnd or bs))
    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)
    print(confusion);

    --local modelId = string.sub(opt.modelPath,-12,-5):gsub('\\','');
    fd = io.open(paths.concat(opt.save, 'confusion-'..modelId..'.txt'), 'w')
    fd:write('Confusion matrix results: \n');
    fd:write('Model:'..opt.modelPath..'\n');
    fd:write('Validation Set:'..opt.valId..'\n')
    fd:write('[1][1]:'..confusion.mat[1][1]..'|[1][2]'..confusion.mat[1][2].."|[2][1]"..confusion.mat[2][1].."|[2][2]"..confusion.mat[2][2]..'\n');
    fd:close()
    confusion:zero()

    result = {output = allOutputs, label = provider.dataset.testData.labels}
    h5 = hdf5.open( paths.concat(opt.save, 'voxel_prediction-'..modelId..'.h5'), 'w')
    h5:write('/result', result )
    h5:close()



end



test()

