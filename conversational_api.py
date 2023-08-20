from pydantic import BaseModel
from fastapi import HTTPException, FastAPI, Response, Depends
from uuid import UUID, uuid4

from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

import conversational_agent
from langchain.agents.agent import AgentExecutor
from typing import Dict

server_storage = {}

class SessionData(BaseModel):
    username: str
    # qa_object_handle : str
    # qa: AgentExecutor
    # qa = conversational_agent.conversatioalAgentsChatGPT().getConversational()

class Conversation(BaseModel):
    question: str
    history: str | None = None

cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)
backend = InMemoryBackend[UUID, SessionData]()


class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[UUID, SessionData],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True


verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

app = FastAPI()


@app.post("/create_session/{name}")
async def create_session(name: str, response: Response):

    session:UUID = uuid4()
    # qa_object_handle=uuid4()

    # Store the object in a server-side dictionary or cache using the handle as the key
    server_storage[session] = conversational_agent.conversatioalAgentsChatGPT().getConversational()

    # data = SessionData(username=name)
    data = SessionData(username=name)


    await backend.create(session, data)
    cookie.attach_to_response(response, session)

    return { "status":f"created session for {name}","session_id" : session}



# @app.get("/whoami/{session_uid}", dependencies=[Depends(cookie)])
# async def whoami(session_uid:UUID = None,session_data: SessionData = Depends(verifier)):
#     print("session_uid",session_uid,"session_data",session_data)
#     if session_uid == None :
#         return session_data
#     else:
#         return session_uid


@app.get("/whoami/{session_uid}")
async def whoami(session_uid:UUID = None):
    print("session_uid",session_uid,"session_data")
    return session_uid


@app.get("/")
async def help():
    return "use /docs for swagger"

@app.get("/sessions")
async def sessions_list():
    keys_and_types = {}
    for key, value in server_storage.items():
        object_type = type(value).__name__
        keys_and_types[key] = object_type
    return keys_and_types

@app.post("/delete_session/{session_uid}")
async def del_session(response: Response, session_uid:UUID = None):
    print("session_uid",session_uid)
    await backend.delete(session_uid)
    cookie.delete_from_response(response)
    # Delete any server-side storage related to the session
    object_handle = session_uid
    if object_handle:
        # Delete the object from the server-side storage
        del server_storage[object_handle]

    return "deleted session"

@app.post("/conversation/{session_uid}")
async def say(conversation:Conversation,session_uid:UUID = None):
    print("session_uid",session_uid)
    object_handle = session_uid
    print("object_handle",object_handle)
    qa =server_storage[object_handle]
    print("object_handle 2")
    res=qa.run(input=conversation.question)
    print("res",res)
    return {"question":conversation.question,"answer":res}