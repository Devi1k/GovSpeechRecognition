run:
	python3 manage.py runserver 10.13.56.38:8000

generate:
	python3 manage.py makemigrations && python3 manage.py migrate