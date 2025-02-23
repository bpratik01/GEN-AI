from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

writer = Agent(
    role='Content Writer',
    goal='Draft a blog post about the benefits of meditation.',
    backstory='You are a content writer with a knack for creating engaging and informative blog posts.',
    llm=llm
)

reviewer = Agent(
    role='Content Reviewer',
    goal='Refine the blog post for clarity and quality.',
    backstory='You are a content reviewer with a keen eye for detail and a passion for quality writing.',
    llm=llm
)

crew = Crew(
    agents=[writer, reviewer],
    tasks=[
        Task(
            description='Draft a blog post about the benefits of meditation.',
            agent=writer,
            expected_output='A well-structured blog post detailing the benefits of meditation.'
        ),
        Task(
            description='Refine the blog post for clarity and quality.',
            agent=reviewer,
            expected_output='A polished, high-quality blog post ready for publishing.'
        )
    ],
    process='Sequential'
)

print('Final Output:')
print(crew.kickoff())
print('---')
