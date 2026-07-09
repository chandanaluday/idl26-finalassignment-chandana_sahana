# Audit Log — Operation Cyber-Histology

This pipeline had a bunch of bugs stacked on top of each other. Here's a list of each one we found. Below is what the code looked like before and after the fix.

---

### 1. Data leak between train and validation sets (data.py, commit 66e4861)

Validation accuracy was way too good, which is usually a sign
something's wrong rather than something to celebrate. The training slice
wasn't actually excluding the validation portion --- both were pulling
from the full array, so the model was quietly being tested on data it
had already seen.

**Before:** train_data = full_data (no exclusion of validation portion)

**After:** train_data = full_data\[:val_start\]

### 2. Label shape mismatch (data.py, commit 66e4861)

Right after that, CrossEntropyLoss started acting up --- crashing or
misbehaving depending on the run. Labels were shaped (N,1) when (N,) was
expected.

**Before:** labels used as-is with shape (N,1)

**After:** labels.squeeze(1) applied to train/val/test labels, giving
shape (N,)

### 3. VGG16 channel mismatch (models.py, commit a6b508f)

VGG16 kept throwing a channel-mismatch error. VGGBlock wasn't updating
current_in_channels between conv layers within the same block, so
channel counts drifted out of sync.

**Before:** current_in_channels left unchanged across conv layers in the
same block.

**After:** current_in_channels = out_channels added at the end of each
loop iteration.

### 4. VGG16 tail layer padding (models.py, commit a6b508f)

The "Config C" tail layer was throwing off spatial dimensions and
breaking things further down the network. It was using the same padding
as the regular 3x3 layers instead of 0 for its 1x1 conv.

**Before:** conv_padding = padding (same padding used for all conv
layers, including the 1x1 tail)

**After:** conv_padding = 0 if is_config_c_tail else padding

### 5. AlexNet hardcoded channels/classes (models.py, commit a6b508f)

AlexNet flat-out refused to run on any non-RGB dataset (chest, orgs),
and gave the wrong number of output classes elsewhere too. in_channels
and num_classes were hardcoded to 3 and 11 --- never actually pulled
from kwargs.

**Before:** in_channels = 3, num_classes = 11 (hardcoded inside
**init**(self, \*\*kwargs))

**After:** in_channels and num_classes added as explicit parameters,
read from kwargs and used throughout the class

### 6. Classifier input size assumption (models.py, commit a6b508f)

AlexNet and VGG16 both crashed with shape mismatches in the classifier
layer. The first Linear layer assumed one specific flattened size
(2048), tied to a single input resolution.

**Before:** nn.Linear(2048, ...) directly after flattening, with no
pooling step

**After:** nn.AdaptiveAvgPool2d((1,1)) added before flatten; Linear
layer's input size corrected to match the actual channel count

### 7. ResNet18 missing activation (models.py, commit a6b508f)

ResNet18 basically wasn't learning anything --- no non-linearity at all,
so it was acting like a glorified linear layer. The activation was being
pulled from a hardcoded global set to "Identity" instead of from kwargs.

**Before:** activation_str = "Identity" (hardcoded module-level global)

**After:** activation_str = kwargs.get("activation_str", "ReLU") read
inside **init**

### 8. ResNet18 missing return statement (models.py, commit a6b508f)

On top of that, ResNet18 threw a TypeError during loss computation.
forward() was computing the output but never actually returning it ---
so it silently returned None.

**Before:** self.classifier(out) computed but not returned (implicit
return None)

**After:** return self.classifier(out)

### 9. Optimizer not resetting gradients (fit.py, commit 3aa7c42)

Loss went straight to NaN within the first epoch --- a classic sign of
exploding gradients. optimizer.zero_grad() was never being called, so
gradients kept piling up across batches instead of resetting.

**Before:** training loop went straight from forward pass to .backward()
with no gradient reset

**After:** self.optimizer.zero_grad() added before each batch's
forward/backward pass

### 10. Shadowed built-in variable (fit.py, commit 3aa7c42)

Variable named sum was quietly shadowing Python's built-in sum().

**Before:** sum = 0 used as a running total variable name

**After:** renamed to total = 0

### 11. Filename pattern mismatch (data.py, commit dc6ee9d)

Every single dataset load failed with FileNotFoundError. The code was
looking for files named {data}\_data.pt, but the actual files were just
{data}.pt.

**Before:** filename = f"{data}\_data.pt"

**After:** filename = f"{data}.pt"
