# Using Nexus Devices Without a Browser

This repository contains examples of how to access Quantinuum Nexus devices programmatically, without opening a browser.

## Install Dependencies

Install the required Python packages:

```bash
pip install python-dotenv qnexus pytket
```

## Files

- **show_devices.py** lists all available devices.  
- **vqe.py** demonstrates an example from the [tutorial](https://docs.quantinuum.com/nexus/trainings/notebooks/knowledge_articles/vqe_example.html).  
- **Credentials** are stored in a **.env** file. Example:

```env
HQS_USERNAME="email@domain.com"
HQS_PASSWORD="password"
```
