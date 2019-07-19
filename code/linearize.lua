require 'torch'
require 'nn'

--[[
This script tests the idea of extracting rules from an NN
by using PReLUs and moving their parameters during training
to make the degenerate into an idenitiy.

Here we train on an artificial dataset.
]]

--[[
ABOUT PRELUs

See
https://github.com/torch/nn/blob/master/doc/transfer.md#prelu
https://github.com/torch/nn/blob/master/PReLU.lua

]]

-- Configuration
local DType = 'torch.FloatTensor'
local InputSize = 4
local OutputSize = 3
local HiddenSize = 5 -- size of hidden layers
local HiddenLayers = 4 -- number of hidden layers
local OptimState = {learningRate = 0.01} -- 0.01
local Problem = 'xor' -- 'xor' or 'addition'
local BatchSize = 4
local AddNoise = true -- whether to add Guassian noise (new noise in every batch)
local NoiseRate = 0.01

-- Keep the network layers for processing
local prelus = {}
local linears = {}

-- Build the model
local function createModelCriterion(inputsize, outputsize)

   local s = nn.Sequential()
   local m

   m = nn.Linear(inputsize, HiddenSize)
   s:add(m)
   table.insert(linears, m)

   for i = 1,HiddenLayers do
      
      m = nn.PReLU(HiddenSize)
      s:add(m)
      table.insert(prelus, m)

      local outsize = HiddenSize
      if i == HiddenLayers then outsize = outputsize end
      m = nn.Linear(HiddenSize, outsize)
      s:add(m)
      table.insert(linears, m)

   end
      
   return s:type(DType), nn.MSECriterion():type(DType)
end

-- Build training data
local function createSample()
   local x = torch.Tensor(16*BatchSize, InputSize)
   local y = torch.Tensor(16*BatchSize, OutputSize)
   local i = 1
   for batch = 1, BatchSize do
      for b1 = 0,1 do
         for b2 = 0,1 do
            for b3 = 0,1 do
               for b4 = 0,1 do
                  -- inputs
                  x[i][1] = b1
                  x[i][2] = b2
                  x[i][3] = b3
                  x[i][4] = b4
                  
                  -- targets
                  
                  if Problem == 'xor' then
                     y[i][1] = (b1 + b2) % 2 -- xor
                     y[i][2] = (b3 + b4) % 2 -- xor
                     y[i][3] = (b1 * b2) -- and
                  end
                  
                  if Problem == 'addition' then
                     local yy = b1*2 + b2 + b3*2 + b4 -- addition of two 2bit numbers
                     y[i][3] = yy % 2
                     yy = math.floor(yy/2)
                     y[i][2] = yy % 2
                     yy = math.floor(yy/2)
                     y[i][1] = yy
                  end
                  
                  i = i + 1
               end -- b4
            end -- b3
         end -- b2
      end -- b1
   end -- batch

   if AddNoise then
      x:add(torch.randn(16*BatchSize, InputSize) * NoiseRate)
   end
   
   return x:type(DType), y:type(DType)
end

-- -----------------------------------------------------------
-- Convert network parameters to expressions
-- -----------------------------------------------------------

-- Print weights and bias as a linear expression
function print_wb(weight, bias, do_normalise)
   local div = 1
   if do_normalise then div = torch.max(torch.abs(weight)) end
   local o = ""
   for j = 1, weight:size(1) do
      o = o .. string.format("%+1.2f*in%d ", weight[j]/div, j)
   end
   o = o .. string.format("%+1.2f", bias/div)
   return o
end

function print_wbs(weight, bias)
   assert(weight:size(1) == bias:size(1), "mismatch between weight and bias")
   for i = 1, weight:size(1) do
      print(string.format("  out%d = %s", i, print_wb(weight[i], bias[i])))
   end
end

-- Multiply / add the matrices to get linear expressions
function convert_as_linear(uptolevel, decisions)

   local weight
   local bias

   for level = 1, uptolevel do

      if level == 1 then -- we start with layer one
         weight = linears[level].weight:clone()
         bias = linears[level].bias:clone()
      else -- keep multiplying / adding the weights and biases. These operations leave the operands intact
         weight = linears[level].weight * weight
         bias = linears[level].weight * bias + linears[level].bias
      end
      
      -- The output of this layer is modified by nonlinear relus and the decisions
      if level < uptolevel then
         for decix, dec in ipairs(decisions) do
            if dec.level == level and dec.decision == 'neg' then -- in "pos", f(x)=x, so nothing to do
               -- print("Apply decision", decix)
               weight[dec.index]:mul(dec.weight)
               bias[dec.index] = bias[dec.index] * dec.weight
            end
         end
      end
      
   end -- for level

   return weight, bias   
end

local ineq_expr = {pos=">", neg="<"}

-- Extract rules from a network with few nonlinear ReLUs
function extract()
   
   local decisions = {}
   
   -- Check all prelus for non-linear ones
   for prelulevel, prelu in ipairs(prelus) do
      for i = 1, prelu.weight:size(1) do
         if prelu.weight[i] <= 0.9995 then -- if it's not a linear relu
            table.insert(decisions, {level=prelulevel, index=i, weight=prelu.weight[i], decision='neg'})
         end
      end
   end

   if #decisions > 5 then
      print("Too many nonlinear PReLUs: "..#decisions)
      return
   end
   
   -- The decisions may depend on each other!
   -- We use the fact that the decisions are ordered by level
   -- so only later ones can depend on earlier ones
   local dodecisions = true
   while dodecisions do
      
      -- Describe the decisions
      for decix, dec in ipairs(decisions) do
         local decw, decb = convert_as_linear(dec.level, decisions)
         print(string.format(
            "IF %s %s 0  (PReLU #%d on level %d is %s. ln(1-weight)=%.2f)",
            print_wb(decw[dec.index], decb[dec.index], true), 
            ineq_expr[dec.decision],
            dec.index, dec.level, dec.decision, math.log(1-dec.weight)
         ))
      end
      print("THEN")
      local w, b = convert_as_linear(#linears, decisions)
      print_wbs(w, b)
      print("")
      
      -- Move on to the next decision pattern (modify the last one first!)
      local addlevel = #decisions -- level of carry
      while decisions[addlevel].decision == 'pos' do
         addlevel = addlevel - 1
         if addlevel < 1 then
            dodecisions = false
            break
         end
      end
   
      if dodecisions then
         for i = #decisions, addlevel+1, -1 do
            decisions[i].decision = 'neg'
         end
         decisions[addlevel].decision = 'pos'
      end
      
   end
   
end


-- -----------------------------------------------------------

local model, criterion = createModelCriterion(InputSize, OutputSize)
local x, y = createSample() -- generate training data

model:training()

local rep = 0
local params, gradParams = model:getParameters()
local adjustrate = 0

-- Pull the parameter of the PReLUs to 1
function adjust_prelus()
   for _, relu in ipairs(prelus) do
      relu.weight = relu.weight + (1-relu.weight):sign() * adjustrate
      -- Limit the weight to be between 0 and 1
      relu.weight:clamp(0, 1)
   end
end

-- Print the parameters of the network
-- We assume that PReLUs are sandwiched between the linear layers
function print_params()
   for lix, m in ipairs(linears) do
      print(string.format("Linear layer #%d:", lix))
      for i = 1, m.weight:size(1) do
         for j = 1, m.weight:size(2) do
            io.write(string.format(" %+1.5f", m.weight[i][j]))
         end
         io.write(" bias:")
         io.write(string.format(" %+1.5f", m.bias[i]))
         io.write("\n")
      end
      if lix <= #prelus then
         io.write("PReLU parameters:")
         relu = prelus[lix]
         for i = 1, relu.weight:size(1) do
            io.write(string.format(" %+1.5f", relu.weight[i]))
         end
         io.write("\n")
      end
   end
end

-- Called by the training function
function evaluate(train_err, train_prediction)
   rep = rep + 1
   if rep < 1000 then return end
   rep = 0
   
   -- Adjust parameters
   -- Increase the prelu adjustment rate as the error falls
   adjustrate = math.max(0,-(math.log(train_err)/math.log(10))-2) * 0.01 * OptimState.learningRate

   print("")
   print("Train error", train_err)
   print("Learning rate", OptimState.learningRate)
   print("PReLU adjust rate", adjustrate)
   print_params()
   extract()

   -- model:evaluate()
   -- model:training()
end

-- The training function
function feval(params)
   gradParams:zero()
   local pred = model:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   model:backward(x, gradCriterion)
   
   -- Do manual training instead of optim
   -- As optim code gets confused if we manually adjust the PReLU parameters
   model:updateParameters(OptimState.learningRate)
   model:zeroGradParameters()
   
   evaluate(err, pred)

   return err, gradParams
end

-- The training loop
while true do
   feval()
   adjust_prelus()

   if AddNoise then
      x, y = createSample() -- get new noise by regenerating training data
   end
end
