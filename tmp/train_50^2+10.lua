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
dofile './provider_50^2+10.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "training/50^2+10")      subdirectory to save logs
   -b,--batchSize             (default 45)          batch size
   -r,--learningRate          (default 0.1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.09)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default model_50^2+10)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   -i,--log_interval          (default 5)           show log interval
   --modelPath                (default training/model.net) exist model
]]

print(opt)
------------------------------------ loading data----------------------------------------
print(c.blue '==>' .. ' loading data')
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

if modelPath and paths.filep(modelPath) then
    model:add(torch.load(modelPath));
    print('==> load exist model:' .. modelPath);
else
    model:add(dofile(opt.model .. '.lua'):cuda());
end

model:get(1).updateGradInput = function(input) return end

----------------------------------- load exist model -------------------------------------





if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model:get(3), cudnn)
end

print(model);


parameters, gradParameters = model:getParameters()

------------------------------------ save log----------------------------------------
print('Will save at ' .. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames { '% mean class accuracy (train set)', '% mean class accuracy (test set)' }
testLogger.showPlot = false

------------------------------------ set criterion---------------------------------------
print(c.blue '==>' .. ' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()

confusion = optim.ConfusionMatrix(2);
------------------------------------ optimizer config-------------------------------------
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}

function train()
    model:training();
    epoch = epoch or 1;

    -- drop learning rate every "epoch_step" epochs  ?
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate / 2 end

    -- update negative set every 6 epochs.
    if epoch % 6 == 0 then
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

    confusion:updateValids();
    print(c.red('Train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)));


    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1;
end




function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print(c.blue '==>' .. " testing")
    local bs = 125
    len = provider.dataset.testData.data:size(1);
    for i = 1, len, bs do
        xlua.progress(i, len)
        if (i + bs) > len then idxEnd = len - i; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.dataset.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
        local outputs = model:forward(provider.dataset.testData.data:narrow(1, i, idxEnd or bs))
        confusion:batchAdd(outputs, provider.dataset.testData.labels:narrow(1, i, idxEnd or bs))
    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)

    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add { train_acc, confusion.totalValid * 100 }
        testLogger:style { '-', '-' }
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

    -- save model every 5 epochs
--    if epoch % 5 == 0 then
        local filename = paths.concat(opt.save, 'model_'..epoch..'.net')
        print('==> saving model to ' .. filename)
        torch.save(filename, model:get(3):clearState())
--    end

    confusion:zero()
end


for i = 1, opt.max_epoch do
    train()
    if epoch % 5 == 0 then
        test()
    end

end

