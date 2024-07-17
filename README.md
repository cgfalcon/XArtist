# XArtist
An virtual artist driven by AI


# Structure of this project

- backend. The flask module of our product.
- frontend. Mainly the React page of the product.
- training. Focus on model traning and testing.

# Backend Installation

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
```

and run the development server

```shell
python -m flask run
```

# FrontEnd Installation

```
npm install next@latest react@latest react-dom@latest
npm install tailwindcss@latest
npm install @headlessui/react @heroicons/react
npm i @material-tailwind/react
npm i prop-types
npm install d3
npm install axios
```


# How to open the webpage?

Go to the `xartist-frontend` directory by the following command

```shell
cd xartist-fronted
```

and start the webserver in develop mode

```shell
npm run dev
```