from pydantic import BaseModel,EmailStr,Field
from typing import Annotated, Optional, Literal

class Person(BaseModel):
    name: str
    age: int
    email: Optional[EmailStr] = None
    cgpa: Optional[float] = Field(2.0, ge=0.0, le=4.0, description="The CGPA of the person, which must be between 0.0 and 4.0.")
    role: Annotated[Literal['admin', 'user', 'guest'], "The role of the person in the system, which can be either 'admin', 'user', or 'guest'."]
    
new_person = Person(name="Alice", age=30, email="alice@example.com",cgpa=3.5, role="user")

person_json= new_person.model_dump_json()

# print(new_person)
print(new_person.name)
print(new_person.age)
print(new_person.email)
print(new_person.role)