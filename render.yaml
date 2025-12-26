services:
  - type: web
    name: crypto-trading-api
    runtime: python
    plan: free
    region: oregon
    branch: main
    
    buildCommand: |
      pip install --upgrade pip
      pip install -r requirements.txt
      echo "=== Structure ==="
      find . -name "*.py" | sort
      echo "=== Python Test ==="
      python -c "
      import sys, os
      print('Python:', sys.version)
      print('CWD:', os.getcwd())
      print('Files:', os.listdir('.'))
      if os.path.exists('api'):
          print('API files:', os.listdir('api'))
      sys.path.insert(0, '.')
      sys.path.insert(0, 'api')
      try:
          import api.main
          print('✅ api.main import OK')
      except Exception as e:
          print(f'❌ api.main: {e}')
      try:
          import main
          print('✅ main import OK')
      except Exception as e:
          print(f'❌ main: {e}')
      "
    
    startCommand: python runner.py
    
    envVars:
      - key: PORT
        value: 10000
      - key: PYTHONUNBUFFERED
        value: "1"
    
    healthCheckPath: /api/health