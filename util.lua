require 'cunn'
local ffi=require 'ffi'


function getMemStats()
    local nGPU = cutorch.getDeviceCount()
    local totalFree =0;
    local total = 0;
    for i=1,nGPU do
        local free1,tot1 = cutorch.getMemoryUsage(i);
        print(string.format('free%d,total%d,used%d Mbs,percentage:%.4f',free1,tot1,torch.round((tot1-free1)/1000000),(tot1 - free1)/tot1));
        totalFree =  totalFree+free1;
        total = total + tot1;
    end
    print(string.format('total free:%d,total:%d. used:%d percentage:%.4f',totalFree,total,torch.round((total - totalFree)/1000000),(total - totalFree)/total));
end

function makeDataParallel(model, nGPU)
    if nGPU > 1 then
        print('converting module to nn.DataParallelTable')
        assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
            cutorch.setDevice(i)
            model:add(model_single:clone():cuda(), i)
        end
    end

    cutorch.setDevice(1)

    return model
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(opt.GPU)
    newDPT:add(module:get(1), opt.GPU)
    return newDPT
end

function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(filename, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(filename, temp_model)
    else
        error('This saving function only works with Sequential or DataParallelTable modules.')
    end
end

function loadDataParallel(filename, nGPU)
    if opt.backend == 'cudnn' then
        require 'cudnn'
    end

    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            print(module);
            if torch.type(module) == 'nn.DataParallelTable' then
                print('do pt');
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
            end
        end
        return model
    else
        error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end