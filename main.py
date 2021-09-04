from fastapi import FastAPI
from src.dtypes import AddIn
from src.dtypes import SearchIn
from src.dtypes import ExistsIn
from src.dtypes import DeleteIn


app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'hello world'}


# TODO: need multiple images here
@app.post('/add')
async def add(
        request: AddIn,
):
    return {'request': request}


@app.post('/search')
async def search(
        request: SearchIn,
):
    return {'request': request}


@app.post('/exists')
async def exists(
        request: ExistsIn,
):
    return {'request': request}


@app.post('/delete')
async def delete(
        request: DeleteIn,
):
    return {'m': request}


# TODO: delete redundant requirements
