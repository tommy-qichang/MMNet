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
require 'mattorch'
local matio = require 'matio'
require 'xlua';
require 'hdf5'
dofile './provider_25^3_fcnn_classification.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "training/25^3")      subdirectory to save logs
   -b,--batchSize             (default 2)          batch size
   -r,--learningRate          (default 0.1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.09)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default train_25^3)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   -i,--log_interval          (default 5)           show log interval
   --modelPath                (default training/model.net) exist model
   --multiGPU                 (default true)    if it's multiGPU
   --balanceWeight              (default 1)     criterion balance weight
   --testFolder                 (default img_1)     test folder name
   --scanZNumber                (default 56)    DICOM zIdx number
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
    model:add(torch.load(modelPath));
    print('==> load exist model:' .. modelPath);
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
testLogger:setNames {  'train 1st acc ', 'train 2ed acc' , 'test 1st acc', 'test 2ed acc'}
testLogger.showPlot = false

------------------------------------ set criterion---------------------------------------
print(c.blue '==>' .. ' setting criterion')
--require('mobdebug').start(nill,8222)
criterion = nn.CrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()
--criterion = cudnn.VolumetricCrossEntropyCriterion(torch.Tensor{1,opt.balanceWeight}):cuda()

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


local function flat3DMatrix(matrix)
    local dim = matrix:nDimension();
    local size = matrix:size();
    local batchSize = 1;
    for i=1,(dim-2) do
        batchSize = batchSize*size[i]
    end
    return matrix:reshape(batchSize,size[dim-1],size[dim]);

end

local function testOut(inputs,targets,result,type,tidx)
    --    local targets = provider.dataset.testData.label:narrow(1, i, idxEnd or bs):squeeze(2);

    local predictResult = result:select(2,2):csub(result:select(2,1)):squeeze(2);
    local predictResult = flat3DMatrix(predictResult):double();

    local predictTarget = flat3DMatrix(targets):double();

    local origInput = flat3DMatrix(inputs):double();

    jLen = predictResult:size(1);
    paths.mkdir(paths.concat(opt.save,opt.testFolder));
    predictTarget = predictTarget-1;
    --        if i==1 then
    for j = 1,jLen do
        if predictTarget:select(1,j):max() ~= 0 then
            local minValue = -math.min(predictResult:select(1,j):min(),0) + 1;
            image.save(paths.concat(opt.save,opt.testFolder,type..'_'..tidx..'_'..j..'_'..epoch..'_test'..'.png'),image.y2jet(predictResult:select(1,j):add(minValue)));
--            image.save(paths.concat(opt.save,opt.testFolder,type..'_'..tidx..'_'..j..'_'..epoch..'_orig'..'.png'), origInput:select(1,j):squeeze());
            image.save(paths.concat(opt.save,opt.testFolder,type..'_'..tidx..'_'..j..'_'..epoch..'_label'..'.png'),predictTarget:select(1,j):squeeze());

        end
        --            matio.save(paths.concat(opt.save,'imgs',((i-1)*bs+j)..'_test'..epoch..'.mat'),results:select(1,j));
    end
    --        end

end

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
    if epoch % opt.epoch_step == 0 then
        optimState.learningRate = optimState.learningRate / 2
        print(c.blue '==>' .. " decrease LR: " ..optimState.learningRate .. ']')
    end
    -- update negative set every 6 epochs.

    print(c.blue '==>' .. " online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


    local targets = torch.CudaTensor(opt.batchSize,provider.dataset.trainData.label:size(2));

    -- random index and split all index into batches.
    local indices = torch.randperm(provider.dataset.trainData.data:size(1)):long():split(opt.batchSize);
    indices[#indices] = nil;


    local tic = torch.tic();
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)
        collectgarbage();
        local innerTic = torch.tic();
        local inputs = provider.dataset.trainData.data:index(1, v);

--        88*56
        targets:copy(provider.dataset.trainData.label:index(1, v));


        model:zeroGradParameters();

        local outputs = model:forward(inputs:float())

        local flatOutputs = outputs:reshape(outputs:size(1)*opt.scanZNumber,2);
        local flatTargets = targets:reshape(targets:size(1)*opt.scanZNumber);
        local f = criterion:forward(flatOutputs, flatTargets)

        local df = criterion:backward(flatOutputs, flatTargets)
        local backDf = df:reshape(targets:size(1),opt.scanZNumber*2);
        model:backward(inputs, backDf);

        confusion:batchAdd(flatOutputs, flatTargets);


        for pk,pv in pairs(parameters) do
            if pk>=77 then
                local gradParameter = gradParameters[pk];

                cutorch.setDevice(pv:getDevice())
                local feval = function(x)
                    return f,gradParameter;
                end
                optim.sgd(feval, pv, optimStatesTable[pk]);

            end

        end

        cutorch.setDevice(1);


--        if(epoch % 2 == 0) then
--            testOut(inputs,targets,outputs,'train',t);
--
--        end

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
        end


        if t % 100 == 0 then
            plotCost(opt.batchSize);
        end


    end

    confusion:updateValids();
    print(c.red('Train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)));
    print(confusion);

    train_acc = confusion.totalValid * 100

    firstClassAcc = confusion.valids[1];
    secondClassAcc = confusion.valids[2];
    confusion:zero()

    epoch = epoch + 1;
    collectgarbage();


end


function test()

    -- disable flips, dropouts and batch normalization
    model:evaluate()
    collectgarbage();
    print(c.blue '==>' .. " testing")
    local bs = 1
    len = provider.dataset.testData.data:size(1);
    local allTargets = nil;
    local allResults = nil;

    for i = 1, len, bs do
        xlua.progress(i, len)
--        if (i + bs) > len then idxEnd = len - i; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.dataset.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
--        local outputs = model:forward(provider.dataset.testData.data:narrow(1, i, idxEnd or bs))
        local inputs = provider.dataset.testData.data:narrow(1, i, bs);
        local outputs = model:forward(inputs)

        collectgarbage();
        local targets = provider.dataset.testData.label:narrow(1, i, bs):squeeze(2);


        local flatOutputs = outputs:reshape(outputs:size(1)*opt.scanZNumber,2);
        local flatTargets = targets:reshape(targets:size(1)*opt.scanZNumber);

        confusion:batchAdd(flatOutputs,flatTargets);

        collectgarbage();

        if(allResults)then
            allResults = torch.cat(allResults,flatOutputs:double(),1);
            allTargets = torch.cat(allTargets,flatTargets:double(),1);
        else
            allResults = flatOutputs:double();
            allTargets = flatTargets:double();
        end


    end

    local rocData = {};
    rocData['oneresult'] = allResults;
    rocData['onetarget'] = allTargets;

--    local resultPath = paths.concat(opt.save,'rocData','fcnn_rocdata_epo_result'..epoch..'.mat');
--    local targetPath = paths.concat(opt.save,'rocData','fcnn_rocdata_epo_target'..epoch..'.mat');
--    print('=====> save roc data:'..resultPath);
    paths.mkdir(paths.concat(opt.save,'rocData'));

--    matio.save(resultPath,rocData.oneresult);
--    matio.save(targetPath,rocData.onetarget);

--    mattorch.save(resultPath,rocData.oneresult);
--    mattorch.save(targetPath,rocData.onetarget);


    local resultPath = paths.concat(opt.save,'rocData','fcnn_rocdata_epo_result'..epoch..'.h5');
    local targetPath = paths.concat(opt.save,'rocData','fcnn_rocdata_epo_target'..epoch..'.h5');

    local myFile = hdf5.open(resultPath, 'w')
    myFile:write('/result', rocData.oneresult);
    myFile:write('/target', rocData.onetarget);
    myFile:close()

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)
    print(confusion);

    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add { firstClassAcc, secondClassAcc, confusion.valids[1], confusion.valids[2] }
        testLogger:style { '+-', '+-' ,'-', '-'}
        testLogger:plot()

        local base64im
        do
            os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save, opt.save))
            os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save, opt.save))
            local f = io.open(opt.save .. '/test.base64')
            if f then base64im = f:read '*all' end
        end

        local file = io.open(opt.save .. '/report.html', 'w')
        file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save, epoch, base64im))
        for k, v in pairs(optimState) do
            if torch.type(v) == 'number' then
                file:write('<tr><td>' .. k .. '</td><td>' .. v .. '</td></tr>\n')
            end
        end
        file:write '</table><pre>\n'
        file:write(tostring(confusion) .. '\n')
        file:write(tostring(model) .. '\n')
        file:write '</pre></body></html>'
        file:close()
    end

    rocData = nil;
    collectgarbage();
    print('=====> finish saving prediction,start saving model...');
    -- save model every 5 epochs
--    if epoch % 5 == 0 then
        local filename = paths.concat(opt.save, 'model_'..epoch..'.net')
        print('==> saving model to ' .. filename)
        torch.save(filename, model:get(2):clearState())
--    end

    confusion:zero()
end

for i = 1, opt.max_epoch do
    train()
    if epoch % 4 == 1 then
        test()
    end
end

