--
-- Description: ${DESC}
-- User: Qi Chang(tommy) <tommy.qichang@gmail.com>
-- Date: 3/16/16
-- Time: 10:35 AM
-- 
--CUDA_VISIBLE_DEVICES=2 th -i predict_partialData.lua --provider=training/provider_voxel_60^2+5.t7 --modelPath=training/60^2+5_r0.001/model_%d.net --startIter=5 --endIter=40

require 'nn';
require 'optim'
require 'cunn'
require 'cudnn'
require 'image';
require 'xlua';
require 'mattorch'

opt = lapp [[
   -p,--provider                  (default "training/provider_voxel.t7")      where test data stored
   -m,--modelPath             (default 'training/25^3_i2/model_%d.net')          where model paremeters stored
   -s,--savePath        (default 'preprocessing/results/test') where roc datasets stored.
   -e,--endIter   (default 35) iteration end at.
   --startIter  (default 5) iteration start at.
   -i,--interval    (default 5) interval of iteration.
]]

print(opt)

--dofile './provider_new.lua'
provider = torch.load(opt.provider);

modelPath = opt.modelPath;


function predictForOneModel(modelPath,provider)
    confusion = optim.ConfusionMatrix(2);

    w1,w2 = string.match(modelPath,'/(.+)/(.+).net')



    model = torch.load(modelPath)

    provider = provider.dataset;
    model:evaluate();
    provider.testData.data = provider.testData.data;
    len = provider.testData.data:size(1);
    bs = 200;

    score = torch.Tensor(len);

    for i = 1, len, bs do
        xlua.progress(i, len)
        if (i + bs) > len then idxEnd = len - i; end
        local outputs = model:forward(provider.testData.data:narrow(1, i, idxEnd or bs):cuda());
        local batchData = outputs:select(2, 2) - outputs:select(2, 1);
        score[{ { i, (i + (idxEnd or bs) - 1)} }] = batchData:float();
        confusion:batchAdd(outputs, provider.testData.labels:narrow(1, i, idxEnd or bs))



    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100);
    print(confusion)

    results = {};
    results['score'] = score;
    results['label'] = provider.testData.labels;


    mattorch.save(opt.savePath .. '/rocdata_'.. w1 .. '_' .. w2..'.mat',results);

    confusion:zero();
end



for i=opt.startIter,opt.endIter,opt.interval do
    print('predict for model:i=%d \n',i);
    realPath = string.format(modelPath,i);

    predictForOneModel(realPath,provider);
end



