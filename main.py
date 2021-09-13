from fastapi import FastAPI
from src.dtypes import AddIn
from src.dtypes import SearchIn
from src.dtypes import ExistsIn
from src.dtypes import DeleteIn
from src.bl import BusinessLogic


app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'hello world'}


@app.post('/add')
async def add(
        request: AddIn,
):
    return BusinessLogic.add(request)


@app.post('/search')
async def search(
        request: SearchIn,
):
    return BusinessLogic.search(request)


@app.post('/exists')
async def exists(
        request: ExistsIn,
):
    return BusinessLogic.exists(request)


@app.post('/delete')
async def delete(
        request: DeleteIn,
):
    return BusinessLogic.delete(request)
