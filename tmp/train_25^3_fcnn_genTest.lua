--
--
-- User: changqi
-- Date: 3/14/16
-- Time: 12:25 PM
-- To change this template use File | Settings | File Templates.
require 'nn';
require 'optim'
require 'cunn'
require 'cudnn'
require 'image';
require 'xlua';
local matio = require 'matio'
dofile './provider_25^3_fcnn.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "training/25^3")      subdirectory to save logs
   -b,--batchSize             (default 45)          batch size
   -r,--learningRate          (default 0.1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.09)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default train_25^3)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   -i,--log_interval          (default 5)           show log interval
   --modelPath                (default training/25^3_i8_r0.005/model_4.net) exist model
   --multiGPU                 (default true)    if it's multiGPU
   --balanceWeight              (default 1)     criterion balance weight
]]

print(opt)
------------------------------------ loading data----------------------------------------
print(c.blue '==>' .. ' loading data')
--provider = torch.load('provider.t7')

------------------------------------ configuring----------------------------------------

print(c.red '==>' .. 'configuring model')
local modelPath = opt.modelPath;

local model = nn.Sequential();

model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));

if modelPath and paths.filep(modelPath) then
    print('==> load exist model:' .. modelPath);
    model:add(torch.load(modelPath)):cuda();
else
    model:add(dofile(opt.model .. '.lua'):cuda());
end




----------------------------------- load exist model -------------------------------------





if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model:get(2), cudnn)
end

print(model);


parameters, gradParameters = model:parameters()

------------------------------------ save log----------------------------------------

print('Will save at ' .. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames { '% mean class accuracy (train set)', '% mean class accuracy (test set)' }
testLogger.showPlot = false

------------------------------------ set criterion---------------------------------------
print(c.blue '==>' .. ' setting criterion')
--criterion = nn.CrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()
criterion = cudnn.VolumetricCrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()

confusion = optim.ConfusionMatrix(2);
------------------------------------ optimizer config-------------------------------------
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}

local pLen = #parameters
local function makeOptimStatesTable (opt)
    local t = {}
    for k = 1, pLen do
        t[k] = tablex.deepcopy(opt)
    end
    return t;
end
optimStatesTable = makeOptimStatesTable(optimState);

cost = {}

function plotCost(avgWidth)
    if not gnuplot then
        require 'gnuplot'
    end
    local avgWidth = avgWidth or 50
    local costT = torch.Tensor(cost)
    local costX = torch.range(1, #cost)
    --
    local nAvg = (#cost - #cost%avgWidth)/avgWidth
    local costAvg = torch.Tensor(nAvg)
    local costAvgX = torch.range(1, nAvg):mul(avgWidth)

    for i = 1,nAvg do
        costAvg[i] = costT[{{(i-1)*avgWidth+1, i*avgWidth}}]:mean()
    end
    plots = {costT, costAvg }
    gnuplot.epsfigure(paths.concat(opt.save,'fcnn_train.eps'));
    gnuplot.plot({'Mini batch cost',costX, costT},
        {'Mean over ' .. avgWidth .. ' batches', costAvgX, costAvg})
    gnuplot.plotflush();
end

function train()

    model:training();
    epoch = epoch or 1;

    -- drop learning rate every "epoch_step" epochs  ?
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate / 2 end

    -- update negative set every 6 epochs.

    print(c.blue '==>' .. " online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


    local targets = torch.CudaTensor(opt.batchSize,provider.dataset.trainData.data:size(3),provider.dataset.trainData.data:size(4),provider.dataset.trainData.data:size(5));
    -- random index and split all index into batches.
    local indices = torch.randperm(provider.dataset.trainData.data:size(1)):long():split(opt.batchSize);
    indices[#indices] = nil;


    local tic = torch.tic();
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)
        collectgarbage();
        local innerTic = torch.tic();
        local inputs = provider.dataset.trainData.data:index(1, v);
        targets:copy(provider.dataset.trainData.label:index(1, v));


        model:zeroGradParameters();
--        require('mobdebug').start(nill,8222)

        local outputs = model:forward(inputs:float())
        local f = criterion:forward(outputs, targets)

        local df = criterion:backward(outputs, targets)
        model:backward(inputs, df)



        for pk,pv in pairs(parameters) do
            if pk>52 then
                local gradParameter = gradParameters[pk];
    --            local flattenParameter = v:view(v:nElement());
    --            local flattenGradParameter = gradParameter:view(gradParameter:nElement());

                cutorch.setDevice(pv:getDevice())
                local feval = function(x)
    --                optimState.dfdx = nil;
                    return f,gradParameter;
                end
                optim.sgd(feval, pv, optimStatesTable[pk]);

            end

        end

        cutorch.setDevice(1);

        table.insert(cost,f)

        local innerToc = torch.toc(innerTic);
        local function printInfo()
            local tmpl = '---------%d/%d (epoch %.3f), ' ..
                    'train_loss = %6.8f, ' ..
                    'speed = %5.1f/s, %5.3fs/iter -----------'
            print(string.format(tmpl,
                t, #indices, epoch,
                f,
                opt.batchSize / innerToc, innerToc))
        end

        if t % opt.log_interval == 0 then
            printInfo();
            plotCost(opt.batchSize);
        end

    end

--    confusion:updateValids();
--    print(c.red('Train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)));


--    train_acc = confusion.totalValid * 100

--    confusion:zero()
    epoch = epoch + 1;
    collectgarbage();


end




function test()

    epoch = 'test'
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    collectgarbage();
    print(c.blue '==>' .. " testing")
    local bs = 2
    len = provider.dataset.testData.data:size(1);
    local allTargets = nil;
    local allResults = nil;
    local allOrigInput = nil;

    for i = 1, len, bs do
        xlua.progress(i, len)
--        require('mobdebug').start(nill,8222)
        if (i + bs) > len then idxEnd = len - i; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.dataset.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
        local inputs = provider.dataset.testData.data:narrow(1, i, idxEnd or bs);
        local outputs = model:forward(inputs)


        collectgarbage();
        local targets = provider.dataset.testData.label:narrow(1, i, idxEnd or bs):squeeze(2);

        local predictResult = outputs:select(2,2):csub(outputs:select(2,1)):squeeze(2);
        predictResult = predictResult:reshape(predictResult:size(1)*predictResult:size(2),
            predictResult:size(3),predictResult:size(4)):double();

        local predictTarget = targets:reshape(targets:size(1)*targets:size(2),
            targets:size(3),targets:size(4)):double();


        local origInput = inputs:squeeze(2);
        origInput = origInput:reshape(origInput:size(1)*origInput:size(2),
            origInput:size(3),origInput:size(4)):double();

        jLen = predictResult:size(1);
        for j = 1,jLen do
            --        require('mobdebug').start(nill,8222);

            local minValue = -math.min(fResult:select(1,j):min(),0) + 1;
            image.save(paths.concat(opt.save,'imgs',(j)..'_test'..epoch..'.png'),image.y2jet(predictResult:select(1,j):add(minValue)));
            image.save(paths.concat(opt.save,'imgs',(j)..'_orig'..epoch..'.png'), origInput:select(1,j):squeeze());
            image.save(paths.concat(opt.save,'imgs',(j)..'_label'..epoch..'.png'),predictTarget:select(1,j):squeeze());


            --            matio.save(paths.concat(opt.save,'imgs',((i-1)*bs+j)..'_test'..epoch..'.mat'),results:select(1,j));
        end


        -------------------save prediction heatmap----------------------

    end
--    matio.save(resultPath,rocData.oneresult);
--    matio.save(targetPath,rocData.onetarget);


--    mattorch.save(result,rocData.oneresult);
--    mattorch.save(target,rocData.onetarget);

    collectgarbage();



    confusion:zero()
end

test()