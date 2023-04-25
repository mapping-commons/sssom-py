# Getting started

- First, create a virtual environment of your choice (anaconda, venv, poetry etc.).
- Install the [cruft](https://github.com/cruft/cruft) package. Cruft enables keeping projects up-to-date with future updates made to this original template.

    ```
    pip install cruft
    ```
- Create a project using the [mapping-commons-cookiecutter](https://github.com/mapping-commons/mapping-commons-cookiecutter) template.
    ```
    cruft create https://github.com/mapping-commons/mapping-commons-cookiecutter
    ```

This kickstarts an interactive session where you declare the following:
- `project_name`: Name of the project. [defaults to: my-commons-name]
- `github_org`: Name of the github org the project belongs to. [defaults to: my-org]
- `project_description`: Description of the project [defaults to: 'This is the project description.']
- `full_name`: Name of the author [defaults to: 'My Name']
- `email`: Author's email [defaults to: 'my-name@my-org.org']
- `yo`: Choose from [1]: Yes, [2]: No [**TEST OPTION FOR NOW**]
- `license`: Choose from [1]: Yes, [2]: No [**TEST OPTION FOR NOW**]