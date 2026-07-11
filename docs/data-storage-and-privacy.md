# Data Storage and Privacy

## What Data Is Stored

FRS stores registered face encodings, not raw face images. A face encoding is a numerical representation generated from a detected face. These encodings are still biometric identifiers and should be handled carefully.

## Storage Location

Registered faces are persisted in:

```text
registered_faces.pkl
```

The file is created in the process working directory when a user successfully registers a face.

## Storage Format

The file is serialized with Python `pickle`. The in-memory structure maps a submitted name to an encoding object:

```python
{
    "Name": {
        "encoding": encoding
    }
}
```

## Security Considerations

- Do not commit `registered_faces.pkl` to source control.
- Do not share the pickle file publicly.
- Restrict filesystem access to the machine running the app.
- Avoid accepting untrusted pickle files because loading pickle data can execute arbitrary code.
- Consider replacing pickle with a safer format or database layer before production use.
- Add authentication before exposing the application outside a trusted local environment.

## Privacy Considerations

Face encodings can identify people and should be treated as sensitive biometric data. Before using the system with real people:

- Get clear consent.
- Explain what is stored and why.
- Provide a way to remove registered identities.
- Define a retention policy.
- Secure backups and exported data.
- Follow applicable privacy laws and organizational policies.

## Data Deletion

To remove all registered faces, stop the application and delete:

```bash
rm registered_faces.pkl
```

To remove one person, a management function or route would need to be added because the current application only supports adding or replacing entries by name.
