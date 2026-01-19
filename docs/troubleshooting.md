# Troubleshooting

Common issues and solutions for illama on Intel Arc GPUs.

## GPU Not Detected

### Symptoms
- `illama doctor` shows Level Zero as NOT FOUND
- Model loading fails with device errors

### Solutions

1. **Install Intel GPU drivers**:
   ```bash
   sudo apt update
   sudo apt install -y intel-gpu-tools level-zero
   ```

2. **Verify GPU visibility**:
   ```bash
   sudo intel_gpu_top
   # Should show GPU activity
   
   ls -la /dev/dri/
   # Should show render* and card* devices
   ```

3. **Check user permissions**:
   ```bash
   # Add user to render and video groups
   sudo usermod -aG render,video $USER
   # Log out and back in
   ```

4. **Docker: Ensure GPU passthrough**:
   ```yaml
   # In docker-compose.yml
   devices:
     - /dev/dri:/dev/dri
   group_add:
     - render
     - video
   ```

## OpenVINO Import Errors

### Symptom
```
ImportError: No module named 'openvino'
```

### Solution
```bash
pip install openvino openvino-genai
```

### Symptom
```
ImportError: No module named 'openvino_genai'
```

### Solution
```bash
pip install openvino-genai
```

## Model Download Failures

### Symptom
```
401 Unauthorized
```

### Solution
Set your HuggingFace token:
```bash
export HF_TOKEN="your-token-here"
# Or
huggingface-cli login
```

### Symptom
```
Access denied. Accept the model terms first.
```

### Solution
1. Go to the model's HuggingFace page
2. Click "Agree and access repository"
3. Wait a few minutes for access to propagate
4. Retry the download

## Model Conversion Failures

### Symptom
```
optimum-cli: command not found
```

### Solution
```bash
pip install "optimum[openvino]"
```

### Symptom
```
Conversion failed: out of memory
```

### Solution
- Try a smaller model
- Close other applications
- Use a machine with more RAM for conversion

## Connection Refused

### Symptom
```
Could not connect to illama-manager: Connection refused
```

### Solutions

1. **Check if server is running**:
   ```bash
   docker ps | grep illama
   # or
   curl http://localhost:11434/health
   ```

2. **Check Docker logs**:
   ```bash
   docker logs illama-manager
   ```

3. **Verify port mapping**:
   ```bash
   docker port illama-manager
   # Should show 11434/tcp -> 0.0.0.0:11434
   ```

## Slow Inference

### Symptoms
- Very low tokens/second
- High latency for first token

### Solutions

1. **Verify GPU is being used**:
   ```bash
   # While running inference
   intel_gpu_top
   # GPU utilization should be high
   ```

2. **Check device setting**:
   ```bash
   # Should be GPU, not CPU
   echo $ILLAMA_DEVICE
   ```

3. **Try INT4 quantization** for better performance

4. **Check for thermal throttling**:
   ```bash
   sudo intel_gpu_top
   # Watch for frequency drops
   ```

## Out of Memory

### Symptom
```
RuntimeError: Failed to allocate memory
```

### Solutions

1. **Use INT4 quantization**:
   ```bash
   illama pull model --weight-format int4
   ```

2. **Reduce max_tokens**:
   ```bash
   illama run model --max-tokens 256 "prompt"
   ```

3. **Ensure one-model policy is enabled**:
   ```bash
   export ILLAMA_ONE_MODEL=1
   ```

4. **Close other GPU applications**

## OpenWebUI Not Showing Models

### Solutions

1. **Verify API URL in OpenWebUI settings**:
   - Settings → Connections
   - OpenAI API Base URL: `http://illama-manager:11434/v1`

2. **Pull models first**:
   ```bash
   illama pull Qwen/Qwen3-8B
   ```

3. **Check API response**:
   ```bash
   curl http://localhost:11434/v1/models | jq
   ```

## Docker Networking Issues

### Symptom
OpenWebUI can't reach illama-manager

### Solutions

1. **Ensure both containers are in same network**:
   ```bash
   docker network ls
   docker network inspect <network>
   ```

2. **Use container name, not localhost**:
   ```
   http://illama-manager:11434/v1  # ✓ Correct
   http://localhost:11434/v1       # ✗ Won't work from another container
   ```

## Getting Help

If you're still having issues:

1. Run `illama doctor` and share the output
2. Check Docker logs: `docker logs illama-manager`
3. Include your hardware info:
   - GPU model
   - Ubuntu version
   - OpenVINO version
