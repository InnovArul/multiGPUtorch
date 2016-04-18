dofile 'models.lua'

opt = {}
opt.GPU = 1
model = createSiameseModel()
input, target = prepareDataForSiameseNetwork()
criterion = nn.ClassNLLCriterion():cuda()

----------------------------------------------------------------------------
--############## Training using single FOR loop #######################
----------------------------------------------------------------------------

print ('training with ' .. table.getn(input) .. ' examples')
local start = os.clock()
for index = 1, table.getn(input) do
  output = model:forward(input[index])
  local err = criterion:forward(output, target[index])
  local t = criterion:backward(output, target[index])
  model:backward(input[index], t)
end
print('For loop training: Elapsed time = ' .. os.clock() - start .. ' seconds')


----------------------------------------------------------------------------
--############### DATAPARALLEL TABLE TRAINING #############################
-----------------------------------------------------------------------------
modelParallel = makeDataParallel(model, 4)
parameters, gradParameters = modelParallel:getParameters()
gradParameters:zero()

local start = os.clock()
output = modelParallel:forward(input)
local err = criterion:forward(output, target)
local t = criterion:backward(output, target)
modelParallel:backward(input, t)
print('DataParallelTable training : Elapsed time = ' .. os.clock() - start .. ' seconds')