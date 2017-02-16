--
-- Description: ${DESC}
-- User: Qi Chang(tommy) <tommy.qichang@gmail.com>
-- Date: 3/16/16
-- Time: 10:35 AM
-- 
--

require 'nn';
require 'optim'
require 'cunn'
require 'cudnn'
require 'image';
require 'xlua';
require 'mattorch'


--dofile './provider_new.lua'

datasetPos = mattorch.load('preprocessing/results/test/img5TestDatasetPos.mat');

datasetNeg = mattorch.load('preprocessing/results/test/img5TestDatasetNeg.mat');

cropImagesPos = datasetPos.cropImagesPos:permute(4,3,2,1);
cropImagesNeg = datasetNeg.cropImagesNeg:permute(4,3,2,1);
posSize = cropImagesPos:size();
negSize = cropImagesNeg:size();
cropImagesPos = cropImagesPos:reshape(posSize[1],1,posSize[2],posSize[3],posSize[4]);
cropImagesNeg = cropImagesNeg:reshape(negSize[1],1,negSize[2],negSize[3],negSize[4]);

origTestSize = posSize[1]+negSize[1];
dataset = {};
dataset.testData = {
    data = torch.FloatTensor(origTestSize, 1, posSize[2], posSize[3], posSize[4]),
    labels = torch.Tensor(origTestSize)
}

dataset.testData.data[{{1,negSize[1]}}] = cropImagesNeg:float();
dataset.testData.data[{{negSize[1]+1, negSize[1]+posSize[1]}}] = cropImagesPos:float();

dataset.testData.labels[{{1,negSize[1]}}] = torch.Tensor(negSize[1]):fill(0):float();
dataset.testData.labels[{{negSize[1]+1, negSize[1]+posSize[1]}}]  = torch.Tensor(posSize[1]):fill(1):float();



local mean = dataset.testData.data:select(2, 1):mean();
local std = dataset.testData.data:select(2, 1):std();

dataset.testData.data:select(2, 1):add(-mean);
dataset.testData.data:select(2, 1):div(std);



modelPath = 'training/25^3_i3_r0.01/model_25.net';

w1,w2 = string.match(modelPath,'/(.+)/(.+).net')

model = torch.load(modelPath)

model:evaluate();
len = dataset.testData.data:size(1);
bs = 256;

score = torch.Tensor(len);

for i = 1, len, bs do
    xlua.progress(i, len)
    if (i + bs) > len then idxEnd = len - i; end
    local outputs = model:forward(dataset.testData.data:narrow(1, i, idxEnd or bs):cuda());
    local batchData = outputs:select(2, 2) - outputs:select(2, 1);
    score[{ { i, (i + (idxEnd or bs) - 1)} }] = batchData:float();
end

results = {};
results['score'] = score;
results['label'] = dataset.testData.labels;

mattorch.save('training/25^3_i2/rocdata_'.. w1 .. '_' .. w2..'.mat',results);