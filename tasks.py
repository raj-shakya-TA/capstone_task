from invoke import task

@task
def code_complexity(c):
    """Run radon complexity check"""
    print("🔍 Running radon code complexity...")
    c.run("radon cc src -s -a", pty=False)  # Use pty=False on Windows

@task
def install(c):
    """Install required dependencies"""
    c.run("pip install -r requirements.txt")

@task
def test(c):
    """Run unit tests"""
    c.run("python -m unittest discover tests")
