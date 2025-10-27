# Custom Model Configuration Guide

## 📦 Your Custom Trained Models

You have successfully configured the following custom trained models:

### 🎯 Vehicle Detection Models
Located in: `models/vehical/`

1. **Vehicle Custom (Best)** - `models/vehical/best.pt`
   - This is the best performing checkpoint from your training
   - Recommended for production use
   - Trained specifically for vehicle detection in your dataset

2. **Vehicle Custom (Last)** - `models/vehical/last.pt`
   - This is the last epoch checkpoint from your training
   - Useful for comparison or if "best" has overfitting issues

### 🌊 Flood/Water Segmentation Models
Located in: `models/flood/`

1. **Flood Custom (Best)** - `models/flood/best.pt`
   - This is the best performing checkpoint from your training
   - Recommended for production use
   - Trained specifically for flood/water segmentation in your dataset

2. **Flood Custom (Last)** - `models/flood/last.pt`
   - This is the last epoch checkpoint from your training
   - Useful for comparison or if "best" has overfitting issues

---

## 🎯 Model Selection in UI

When you run the application, you'll see your custom models at the **top** of the dropdown lists:

### Object Detection Dropdown:
```
🎯 Vehicle Custom (Best)    ← Your custom model (Recommended)
🎯 Vehicle Custom (Last)    ← Your custom model (Alternative)
YOLOv11n
YOLOv11s
YOLOv11m
YOLOv11l
YOLOv8n
YOLOv8s
```

### Semantic Segmentation Dropdown:
```
🌊 Flood Custom (Best)      ← Your custom model (Recommended)
🌊 Flood Custom (Last)      ← Your custom model (Alternative)
YOLOv11n-seg
YOLOv11s-seg
YOLOv11m-seg
YOLOv8n-seg
YOLOv8s-seg
```

---

## 🚀 Recommended Configuration

For **best results with your trained models**:

1. **Detection Model**: Select `🎯 Vehicle Custom (Best)`
2. **Segmentation Model**: Select `🌊 Flood Custom (Best)`
3. **Confidence Threshold**: Start with `0.25`, adjust if needed

This combination uses your specifically trained models that are optimized for:
- Vehicle detection in flood scenarios
- Water/flood segmentation

---

## 📊 When to Use Each Model

### Use Custom Models When:
- ✅ Processing images similar to your training data
- ✅ You need maximum accuracy for your specific use case
- ✅ You've trained on flood-specific imagery
- ✅ For your October 28th demonstration

### Use Official YOLO Models When:
- 🔄 Testing general object detection capability
- 🔄 Your custom models aren't performing well
- 🔄 Processing very different imagery
- 🔄 As a baseline for comparison

---

## 🔍 Model Information

### What is "best.pt" vs "last.pt"?

During YOLO training:

**best.pt**:
- Saved when validation metrics are best
- Usually has the highest mAP (mean Average Precision)
- **Recommended for production**
- Less likely to be overfitted

**last.pt**:
- Saved at the end of training (final epoch)
- May have trained longer
- Good for comparison
- Might be overfitted if trained too long

### Which Should You Use?

**For demonstration**: Start with **best.pt** models
- `🎯 Vehicle Custom (Best)`
- `🌊 Flood Custom (Best)`

**If results aren't good**: Try **last.pt** models or compare both

---

## 🛠️ Testing Your Models

### Quick Test (Recommended)

1. **Run the test script**:
```bash
python test_custom_models.py
```

2. **Or test in the application**:
```bash
streamlit run app.py
```

3. **Select your custom models** from the dropdowns

4. **Upload a test image** and verify results

---

## 📝 Model Training Details

### If you need to know what your models were trained on:

Check your training folder for:
- `data.yaml` - Dataset configuration
- `results.csv` or `results.png` - Training metrics
- Training logs

### Typical Custom Model Classes:

**Vehicle Detection Model** likely detects:
- Cars
- Trucks
- Buses
- Motorcycles
- (Possibly) Submerged vehicles

**Flood Segmentation Model** likely segments:
- Water/Flood regions
- Road/Ground
- Buildings
- Sky
- (Your specific classes)

---

## 🔧 Troubleshooting

### Model Won't Load

**Error**: `FileNotFoundError: models/vehical/best.pt not found`

**Solution**: Verify file exists
```bash
ls models/vehical/
ls models/flood/
```

### Model Loads But Poor Results

**Possible Reasons**:
1. Test images very different from training data
2. Model needs different confidence threshold
3. Image resolution mismatch

**Solutions**:
- Adjust confidence threshold (try 0.1 - 0.5 range)
- Try the "last.pt" version
- Compare with official YOLO models
- Check your training metrics

### Model Compatibility Issues

**Error**: `Model version mismatch` or similar

**Solution**: Ensure you're using the same Ultralytics version that trained the model
```bash
# Check version
pip show ultralytics

# Update if needed
pip install --upgrade ultralytics
```

---

## 📂 File Structure

```
Project3/
└── models/
    ├── vehical/
    │   ├── best.pt          ← Your vehicle detection model (best)
    │   └── last.pt          ← Your vehicle detection model (last)
    └── flood/
        ├── best.pt          ← Your flood segmentation model (best)
        └── last.pt          ← Your flood segmentation model (last)
```

---

## 🎯 For Your October 28th Demo

### Recommended Setup:

1. **Use Custom Models**:
   - Detection: `🎯 Vehicle Custom (Best)`
   - Segmentation: `🌊 Flood Custom (Best)`

2. **Prepare Test Images**:
   - Use images similar to training data
   - Have backup images ready
   - Test beforehand!

3. **Have Comparison Ready**:
   - Show results with custom models
   - Compare with official YOLO models
   - Demonstrate model switching capability

4. **Adjust Settings**:
   - Fine-tune confidence threshold for best visual results
   - Practice the workflow

---

## 💡 Pro Tips

1. **"Best" is usually better than "Last"** - Start there

2. **Lower confidence = more detections** (but more false positives)
   - Floods: Try 0.2-0.3
   - Vehicles: Try 0.25-0.4

3. **Compare models side-by-side**:
   - Process same image with custom and official models
   - Document which performs better

4. **Export your best results** before the demo

5. **Keep official models as backup** in case custom models have issues

---

## ✅ Configuration Complete!

Your custom models are now integrated into the system. The configuration has been updated in:

- ✅ `config.py` - Model paths added
- ✅ `app.py` - Using config for model selection
- ✅ Model files verified in `models/` directory

**You're ready to use your custom trained models!** 🎉

Run the application:
```bash
streamlit run app.py
```

Or test first:
```bash
python test_custom_models.py
```
