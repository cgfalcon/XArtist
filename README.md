# The Anonymous Artist, aka XArtist

'The Anonymous Artist' is an generative art platform, which integrated innovative mouse motion input to enhance user interaction and output intuitiveness.

<img width="1024" alt="Screenshot 2024-07-20 at 20 57 55" src="https://github.com/user-attachments/assets/f70d9492-e353-47e7-a9aa-26360e37b970">
<img width="1024" alt="Screenshot 2024-07-20 at 21 00 50" src="https://github.com/user-attachments/assets/6d98452f-8e75-49d9-adb6-75e701f51c7e">
<img width="1024" alt="Screenshot 2024-07-20 at 21 28 01" src="https://github.com/user-attachments/assets/a880bbab-6040-49a2-b14f-401d5bf9e11f">


# Examples generated from the platform

![af4e2634d163b2c9d7910e7fef26bb07](https://github.com/user-attachments/assets/aa05f470-92a8-493a-b5ce-80547e907c5c)


# Structure of this project

- backend. The flask module of our product.
- frontend. Mainly the React page of the product.
- training. Focus on model traning and testing.

## Backend Installation

Go to the `xartist-backend` directory by the following command

```shell
cd xartist-backend
```

Create and activate virtual environment if you are using VSCode (for example, call it '.your_vitural_environment')

```shell
python -m venv .your_vitural_environment
.your_vitural_environment\Scripts\activate.ps1
```

Install packages for python in the virtual environment listed in 'requirements.txt'

```shell
pip install -r .\requirements.txt
pip install torchvision
pip install ISR --no-deps
pip install tensorflow
pip install matplotlib
pip install opencv-python
```

and run the development server

```shell
python -m flask run
```

## FrontEnd Installation

```
npm install next@latest react@latest react-dom@latest
npm install tailwindcss@latest
npm install @headlessui/react @heroicons/react
npm i @material-tailwind/react
npm i prop-types
npm install d3
npm install axios
```


## How to open the webpage?

Go to the `xartist-frontend` directory by the following command

```shell
cd xartist-fronted
```

and start the webserver in develop mode

```shell
npm run dev
```
