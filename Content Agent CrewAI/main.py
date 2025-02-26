from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

writer = Agent(
    role='Content Writer',
    goal='Draft a blog post about the benefits of staying consistent.',
    backstory='You are a content writer with a knack for creating engaging and informative blog posts.',
    llm=llm
)

reviewer = Agent(
    role='Content Reviewer',
    goal='Refine the blog post for clarity and quality.',
    backstory='You are a content reviewer with a keen eye for detail and a passion for quality writing.',
    llm=llm
)

task1 = Task(
    description='Draft a blog post about the benefits of meditation.',
    agent=writer,
    expected_output='A well-structured blog post detailing the benefits of meditation.'
)

task2 = Task(
    description='Refine the blog post for clarity and quality.',
    agent=reviewer,
    expected_output='A polished, high-quality blog post ready for publishing.'
)

crew = Crew(
    agents=[writer, reviewer],   
    tasks=[task1, task2],       
    verbose=True                
)

print('Final Output:')
print(crew.kickoff())
print('---')
