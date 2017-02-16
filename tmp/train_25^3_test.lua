--
--
-- User: changqi
-- Date: 3/14/16
-- Time: 12:25 PM
-- To change this template use File | Settings | File Templates.
require 'nn';
require 'optim'
require 'cunn'
require 'image';
require 'xlua';
require 'math';
require 'cudnn'
matio = require 'matio'
--dofile './provider_25^3.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "preprocessing/results/test/testFCNN")      subdirectory to save logs
   -b,--batchSize             (default 45)          batch size
   --model                    (default train_25^3)     model name
   --modelPath                (default training/25^3_i6_r0.01_w1/model_4.net) exist model
   --testSource               (default preprocessing/results/test/all_testimg_1127_fcnn_rocSourceData.mat)
   --testLabelSource               (default preprocessing/results/test/all_testlabel_1127_fcnn_neg.mat)
]]

print(opt)
------------------------------------ loading data----------------------------------------
print(c.blue '==>' .. ' loading data')


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

if modelPath and paths.filep(modelPath) then
    model:add(torch.load(modelPath));
    print('==> load exist model:' .. modelPath);
else
    model:add(dofile(opt.model .. '.lua'):cuda());
end

model:get(1).updateGradInput = function(input) return end

----------------------------------- load exist model -------------------------------------



print(model);


parameters, gradParameters = model:getParameters()

------------------------------------ save log----------------------------------------
print('Will save at ' , opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames { 'train 1st acc ', 'train 2ed acc' , 'test 1st acc', 'test 2ed acc'}
testLogger.showPlot = false

------------------------------------ set criterion---------------------------------------
print(c.blue '==>' .. ' setting criterion')
criterion = nn.CrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()

confusion = optim.ConfusionMatrix(2);
------------------------------------ optimizer config-------------------------------------
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}
function Pnormalize(selfv)
    require('mobdebug').start(nill,8222)
    local trainData = selfv:select(2,1);

    collectgarbage();
    print '###> pre-processing data'

    local mean = trainData:mean();
    local std = trainData:std();
    trainData:add(-mean);
    trainData:div(std);
    return selfv;

end


function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print(c.blue '==>' .. " testing")
    local bs = 512
--    require('mobdebug').start(nill,8222)

    local testData = matio.load(opt.testSource,'cropImages');

    testData =testData:reshape(testData:size(1),1,testData:size(2),testData:size(3),testData:size(4)):float();
    Pnormalize(testData);

    len = testData:size(1);
    for i = 1, len, bs do
        xlua.progress(i, len)
        if (i + bs-1) > len then idxEnd = len - i +1; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.dataset.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
        local outputs = model:forward(testData:narrow(1, i, idxEnd or bs))

        if(allResults)then
            allResults = torch.cat(allResults,outputs,1);
        else
            allResults = outputs;
        end

    end

    matio.save(paths.concat(opt.save,'allresult.mat'),allResults:float());


    confusion:zero()
end

test()

