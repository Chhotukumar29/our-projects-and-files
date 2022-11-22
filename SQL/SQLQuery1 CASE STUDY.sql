CREATE DATABASE [INTELLIPAAT ASSIGNMENT]
USE [INTELLIPAAT ASSIGNMENT]

CREATE TABLE [LOCATION]
(
 LOCATION_ID INT PRIMARY KEY ,
 CITY VARCHAR(20)
 )
 INSERT INTO [LOCATION] VALUES
 (112,'NEW YORK'),
 (123,'DALLAS'),
 (124,'CHICAGO'),
 (167,'BOSTON')

 CREATE TABLE DEPARTMENT
 (
  DEPARTMENT_ID INT PRIMARY KEY,
  [NAME] VARCHAR(20),
  LOCATION_ID INT FOREIGN KEY REFERENCES [LOCATION](LOCATION_ID)
  )
  INSERT INTO DEPARTMENT VALUES
  (10,'ACCOUNTING',112),
  (20,'SALES',123),
  (30,'RESEARCH',124),
  (40,'OPERATIONS',167)

CREATE TABLE JOB
(
   [JOB ID] INT PRIMARY KEY,
   DESIGNATION VARCHAR(20)
   )
INSERT INTO JOB VALUES
(667,'CLERK'),
(668,'STAFF'),
(669,'ANALYST'),
(670,'SALES_PERSON'),
(671,'MANAGER'),
(672,'PRESIDENT')

CREATE TABLE EMPLOYEE
(
 EMPLOYEE_ID INT,
 LAST_NAME VARCHAR(20),
 FIRST_NAME VARCHAR(20),
 MIDDLE_NAME CHAR(1),
 JOB_ID INT FOREIGN KEY REFERENCES JOB([JOB ID]),
 MANAGER_ID INT,
 HIRE_DATE DATE,
 SALARY INT,
 COMM INT,
 DEPARTMENT_ID INT FOREIGN KEY REFERENCES DEPARTMENT(DEPARTMENT_ID)
 )
 INSERT INTO EMPLOYEE VALUES
 (7369,'SMITH','JOHN','Q',667,7902,'12-DEC-84',800,NULL,20),
 (7499,'ALLEN','KEVIN','J',670,7698,'20-FEB-84',1600,300,30),
 (7505,'DOYLE','JEAN','K',671,7839,'04-APR-85',2850,NULL,30),
 (7506,'DENNIS','LYNN','S',671,7839,'15-MAY-85',2750,NULL,30),
 (7507,'BAKER','LESLIE','D',671,7839,'10-JUN-85',2200,NULL,40),
 (7521,'WARK','CYNTHIA','D',670,7698,'22-FEB-85',1250,500,30)


 --SIMPLE QUERIES
 --1. LIST ALL THE EMPLOYEE DETAILS
 SELECT * FROM EMPLOYEE
 --2.LIST ALL THE DEPARTMENT DETAILS
 SELECT * FROM DEPARTMENT
 --3.LIST ALL JOB DETAILS
 SELECT * FROM JOB
 --4.LIST ALL THE LOCATIONS
 SELECT * FROM [LOCATION]
 --5.LIST OUT THE FIRSTNAME,LASTNAME,SALARY,COMMISSION FOR ALL EMPLOYEES
 SELECT FIRST_NAME,LAST_NAME,SALARY,COMM FROM EMPLOYEE 
 --6.LIST OUT EMPLOYEE ID,LAST NAME,DEPARTMENT-ID FOR ALL EMPLOYEES AND ALIAS EMPLOYEEID AS " ID OF THE EMPLOYEE",LAST NAME AS " NAME OF THE EMPLOYEE", DEPARTMENT ID AS 'DEP_ID"
 SELECT EMPLOYEE_ID AS 'ID OF THE EMPLOYEE',LAST_NAME AS 'NAME OF THE EMPLOYEE',DEPARTMENT_ID AS DEPT_ID FROM [EMPLOYEE]

 --7. LIST OUT THE EMPLOYEES ANNUAL SALARY WITH THEIR NAMES ONLY.
SELECT SALARY*12 AS ANNUAL_SAL FROM EMPLOYEE


--WHERE CONDITION:

--1. LIST THE DETAILS ABOUT "SMITH"
SELECT * FROM EMPLOYEE WHERE LAST_NAME = 'SMITH'


--2. LIST OUT THE EMPLOYEES WHO ARE WORKING IN DEPARTMENT 20.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = '20'


--3. LIST OUT THE EMPLOYEES WHO ARE EARNING SALARY BETWEEN 3000 AND 4500.
SELECT * FROM EMPLOYEE WHERE SALARY BETWEEN 3000 AND 4500


--4. LIST OUT THE EMPLOYEES WHO ARE WORKING IN DEPARTMENT 10 OR 20.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = 10 OR DEPARTMENT_ID = 20 


--5. FIND OUT THE EMPLOYEES WHO ARE NOT WORKING IN DEPARTMENT 10 OR 30.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = 10 OR DEPARTMENT_ID = 20 


--6. LIST OUT THE EMPLOYEES WHOSE NAME STARTS WITH 'S'.
SELECT * FROM EMPLOYEE WHERE LAST_NAME LIKE 'S%'


--7. LIST OUT THE EMPLOYEES WHOSE NAME STARTS WITH 'S' AND ENDS WITH 'H'.
SELECT * FROM EMPLOYEE WHERE LAST_NAME LIKE 'S%H'


--8. LIST OUT THE EMPLOYEES WHOSE NAME LENGTH IS 4 AND START WITH 'S'.
SELECT * FROM EMPLOYEE WHERE LEN(LAST_NAME)=4 AND LAST_NAME ='S%'


--9. LIST OUT EMPLOYEES WHO ARE WORKING IN DEPARRTMENT 10 AND DRAW THE SALARIES MORE THAN 3500.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = 10 AND SALARY> 3500


--10. LIST OUT THE EMPLOYEES WHO ARE NOT RECEVING COMMISSION.
SELECT * FROM EMPLOYEE WHERE COMM is NULL


--ORDER BY CLAUSE:

--1. LIST OUT THE EMPLOYEE ID, LAST NAME IN ASCENDING ORDER BASED ON THE EMPLOYEE ID.
SELECT EMPLOYEE_ID,LAST_NAME FROM EMPLOYEE ORDER BY LAST_NAME,EMPLOYEE_ID


--2. LIST OUT THE EMPLOYEE ID, NAME IN DESCENDING ORDER BASED ON SALARY.
SELECT EMPLOYEE_ID,CONCAT(FIRST_NAME,' ',LAST_NAME) FROM EMPLOYEE ORDER BY SALARY

--3. LIST OUT THE EMPLOYEE DETAILS ACCORDING TO THEIR LAST-NAME IN ASCENDING ORDER 
SELECT * FROM EMPLOYEE ORDER BY LAST_NAME


--4. LIST OUT THE EMPLOYEE DETAILS ACCORDING TO THEIR LAST-NAME IN ASCENDING ORDER AND THEN ON DEPARTMENT_ID IN DESCENDING ORDER.
SELECT * FROM EMPLOYEE ORDER BY LAST_NAME  
SELECT * FROM EMPLOYEE ORDER BY DEPARTMENT_ID DESC


--GROUP BY & HAVING CLAUSE

--1. HOW MANY EMPLOYEES WHO ARE IN DIFFERENT DEPARTMENTS WISE IN THE ORGANIZATION.
SELECT COUNT(*) AS NO_OF_EMPLOYEES, DEPARTMENT_ID FROM EMPLOYEE E GROUP BY DEPARTMENT_ID


--2. LIST OUT THE DEPARTMENT WISE MAXIMUM SALARY, MINIMUM SALARY, AVERAGE SALARY OF THE EMPLOYEES.
SELECT MAX(SALARY) AS MAX_SALARY, MIN(SALARY) AS MIN_SALARY,AVG(SALARY) AS AVG_SALARY FROM EMPLOYEE GROUP BY DEPARTMENT_ID


--3. LIST OUT JOB WISE MAXIMUM SALARY, MINIMUM SALARY, AVERAGE SALARIES OF THE EMPLOYEES.
SELECT MAX(SALARY) AS MAX_SALARY, MIN(SALARY) AS MIN_SALARY,AVG(SALARY) AS AVG_SALARY FROM EMPLOYEE GROUP BY JOB_ID


--4. LIST OUT THE NUMBER OF EMPLOYEES JOINED IN EVERY MONTH IN ASCENDING ORDER.
SELECT COUNT(*) AS NO_OF_EMPLOYEE,MONTH(HIRE_DATE) AS MONTH_NAME FROM EMPLOYEE GROUP BY MONTH(HIRE_DATE) ORDER BY MONTH(HIRE_DATE) 


--5. LIST OUT THE NUMBER OF EMPLOYEES FOR EACH MONTH AND YEAR, IN THE ASCENDING ORDER BASED ON THE YEAR, MONTH.
SELECT COUNT(*) AS NO_OF_EMPLOYEES,MONTH(HIRE_DATE) AS MONTH_NAME,YEAR(HIRE_DATE) AS YEAR_NO FROM EMPLOYEE
GROUP BY MONTH(HIRE_DATE),YEAR(HIRE_DATE) ORDER BY MONTH(HIRE_DATE), YEAR(HIRE_DATE)


--6. LIST OUT THE DEPARTMENT ID HAVING ATLEAST FOUR EMPLOYEES.
SELECT DEPARTMENT_ID, COUNT(*) AS NO_OF_EMPLOYEES FROM EMPLOYEE GROUP BY DEPARTMENT_ID HAVING COUNT(*) >= 4


--7. HOW MANY EMPLOYEES JOINED IN JANUARY MONTH.
SELECT COUNT(*) AS NO_OF_EMPLOYEES FROM EMPLOYEE GROUP BY MONTH(HIRE_DATE) HAVING MONTH(HIRE_DATE) = '01'
DECLARE @HIRE_DATE DATETIME
SET @HIRE_DATE=GETDATE()
SELECT CONVERT(VARCHAR,@HIRE_DATE,6) as [DD MMM YY]

--8. HOW MANY EMPLOYEES JOINED IN JANUARY OR SEPTEMBER MONTH.
SELECT COUNT(*) FROM EMPLOYEE AS NO_OF_EMPLOYEE GROUP BY MONTH(HIRE_DATE) 
HAVING MONTH(HIRE_DATE) = '01' OR MONTH(HIRE_DATE) = '09'


--9. HOW MANY EMPLOYEES WERE JOINED IN 1985?
SELECT COUNT(*) AS NO_OF_EMPLOYEE FROM EMPLOYEE GROUP BY YEAR(HIRE_DATE) HAVING YEAR(HIRE_DATE) = 1985


--10. HOW MANY EMPLOYEES WERE JOINED EACH MONTH IN 1985.
SELECT COUNT(*) AS NO_OF_EMPLOYEE,MONTH(HIRE_DATE) AS HIRE_MONTH,YEAR(HIRE_DATE) AS HIRE_YEAR FROM EMPLOYEE
 GROUP BY MONTH(HIRE_DATE),YEAR(HIRE_DATE) HAVING YEAR(HIRE_DATE) = 1985

--11. HOW MANY EMPLOYEES WERE JOINED IN MARCH 1985?
SELECT COUNT(*) AS NO_OF_EMPLOYEE,MONTH(HIRE_DATE) AS HIRE_MONTH,YEAR(HIRE_DATE) AS HIRE_YEAR FROM EMPLOYEE
 GROUP BY MONTH(HIRE_DATE),YEAR(HIRE_DATE) HAVING YEAR(HIRE_DATE) = 1985 AND MONTH(HIRE_DATE)= 03

--12. WHICH IS THE DEPARTMENT ID, HAVING GREATER THAN OR EQUAL TO 3 EMPLOYEES JOINED IN APRIL 1985?
SELECT TOP 3 DEPARTMENT_ID,COUNT(*) AS NO_OF_EMPLOYEES FROM EMPLOYEE WHERE MONTH(HIRE_DATE) =04 AND YEAR(HIRE_DATE) = 1985
GROUP BY DEPARTMENT_ID HAVING COUNT(*)>= 3


--JOINS

--1. LIST OUT EMPLOYEES WITH THEIR DEPARTMENT NAMES.
SELECT E.*,D.[NAME] FROM EMPLOYEE E
JOIN DEPARTMENT D 
ON D.DEPARTMENT_ID = E.DEPARTMENT_ID


--2. DISPLAY EMPLOYEES WITH THEIR DESIGNATIONS.
SELECT E.*,J.DESIGNATION FROM EMPLOYEE E
JOIN JOB J
ON E.JOB_ID = J.[JOB ID]


--3. DISPLAY THE EMPLOYEES WITH THEIR DEPARTMENT NAMES AND REGIONAL GROUPS.
SELECT E.*, D.[NAME],L.LOCATION_ID FROM EMPLOYEE E
JOIN DEPARTMENT D
ON E.DEPARTMENT_ID = D.DEPARTMENT_ID
JOIN [LOCATION] L
ON L.LOCATION_ID = D.LOCATION_ID


--4. HOW MANY EMPLOYEES WHO ARE WORKING IN DIFFERENT DEPARTMENTS AND DISPLAY WITH DEPARTMENT NAMES.
 SELECT COUNT(EMPLOYEE_ID) AS NO_OF_EMPLOYEE,D.[NAME] FROM EMPLOYEE E
 JOIN DEPARTMENT D 
 ON D.DEPARTMENT_ID= E.DEPARTMENT_ID GROUP BY D.[NAME]


--5. HOW MANY EMPLOYEES WHO ARE WORKING IN SALES DEPARTMENT.
SELECT COUNT(EMPLOYEE_ID) AS EMP_ID, D.[NAME] FROM EMPLOYEE E 
JOIN DEPARTMENT D 
ON E.DEPARTMENT_ID = D.DEPARTMENT_ID 
WHERE D.[NAME]= 'SALES' GROUP BY D.[NAME]


--6. WHICH IS THE DEPARTMENT HAVING GREATER THAN OR EQUAL TO 5 EMPLOYEES AND DISPLAY THE DEPARTMENT NAMES IN ASCENDING ORDER.
SELECT COUNT(EMPLOYEE_ID) AS EMP_ID, D.[NAME] FROM EMPLOYEE E
JOIN DEPARTMENT D 
ON E.DEPARTMENT_ID = D.DEPARTMENT_ID GROUP BY [NAME] HAVING COUNT(EMPLOYEE_ID)>=5 ORDER BY [NAME] 


--7. HOW MANY JOBS IN THE ORGANIZATION WITH DESIGNATIONS.
SELECT J.DESIGNATION, COUNT(EMPLOYEE_ID) AS JOBS FROM JOB J 
JOIN EMPLOYEE E 
ON E.JOB_ID = J.[JOB ID] GROUP BY DESIGNATION

--8. HOW MANY EMPLOYEES ARE WORKING IN "NEW YORK".
SELECT L.CITY, COUNT(EMPLOYEE_ID) AS NO_OF_EMPLOYEE FROM EMPLOYEE E
JOIN DEPARTMENT D 
ON E.DEPARTMENT_ID = D.DEPARTMENT_ID
JOIN [LOCATION] L ON L.LOCATION_ID = D.LOCATION_ID WHERE CITY = 'NEW YORK' GROUP BY CITY

--9. DISPLAY THE EMPLOYEE DETAILS WITH SALARY GRADES
SELECT *,
CASE 
WHEN SALARY<1000 THEN 'C'
WHEN SALARY<2000 THEN 'B'
WHEN SALARY>2000 THEN 'A'
END AS GRADE
FROM EMPLOYEE


--10. LIST OUT THE NO. OF EMPLOYEES ON GRADE WISE. 
ALTER TABLE EMPLOYEE
ADD  GRADE VARCHAR(1)
UPDATE EMPLOYEE
SET GRADE = (
CASE 
WHEN SALARY<1000 THEN 'C'
WHEN SALARY<2000 THEN 'B'
WHEN SALARY>2000 THEN 'A'
END) 
SELECT COUNT(*) AS NO_OF_EMPLOYEES, GRADE FROM EMPLOYEE GROUP BY GRADE

--11. DISPLAY THE EMPLOYEE SALARY GRADES AND NO. OF EMPLOYEES BETWEEN 2000 TO 5000 RANGE OF SALARY.
SELECT COUNT(*) AS NO_OF_EMPLOYEES, GRADE FROM EMPLOYEE WHERE SALARY BETWEEN 2000 AND 5000 GROUP BY GRADE

--16. DISPLAY ALL EMPLOYEES IN SALES OR OPERATION DEPARTMENTS.
SELECT E.* FROM EMPLOYEE E 
JOIN DEPARTMENT D
ON D.DEPARTMENT_ID = E.DEPARTMENT_ID
WHERE D.[NAME] = 'SALES' OR D.[NAME] = 'OPERATION'



--SET OPERATORS

--1. LIST OUT THE DISTINCT JOBS IN SALES AND ACCOUNTING DEPARTMENTS.
 SELECT DISTINCT DESIGNATION from JOB WHERE [JOB ID] IN(SELECT JOB_ID FROM EMPLOYEE 
 WHERE DEPARTMENT_ID=(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='SALES'))
 UNION 
SELECT DISTINCT DESIGNATION from JOB WHERE [JOB ID] IN(SELECT JOB_ID FROM EMPLOYEE 
WHERE DEPARTMENT_ID=(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='ACCOUNTING'))

--2. LIST OUT ALL THE JOBS IN SALES AND ACCOUNTING DEPARTMENTS.
 SELECT DESIGNATION from JOB WHERE [JOB ID] IN(SELECT JOB_ID FROM EMPLOYEE 
 WHERE DEPARTMENT_ID=(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='SALES'))
 UNION ALL
SELECT DESIGNATION from JOB WHERE [JOB ID] IN(SELECT JOB_ID FROM EMPLOYEE 
WHERE DEPARTMENT_ID=(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='ACCOUNTING'))


--3. LIST OUT THE COMMON JOBS IN RESEARCH AND ACCOUNTING DEPARTMENTS IN ASCENDING ORDER.
SELECT DESIGNATION FROM JOB WHERE [JOB ID]IN(SELECT JOB_ID FROM EMPLOYEE WHERE DEPARTMENT_ID =
(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='RESEARCH'))
INTERSECT
SELECT DESIGNATION FROM JOB WHERE[JOB ID] IN(SELECT JOB_ID FROM EMPLOYEE WHERE DEPARTMENT_ID =
(SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME]='ACCOUNTING'))



--SUB QUERIES

--1. DISPLAY THE EMPLOYEES LIST WHO GOT THE MAXIMUM SALARY.
SELECT * FROM EMPLOYEE WHERE SALARY = (SELECT MAX(SALARY) FROM EMPLOYEE)


--2. DISPLAY THE EMPLOYEES WHO ARE WORKING IN SALES DEPARTMENT.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = (SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME] = 'SALES')


--3. DISPLAY THE EMPLOYEES WHO ARE WORKING AS 'CLERCK'.
SELECT * FROM EMPLOYEE WHERE JOB_ID = (SELECT JOB_ID FROM JOB WHERE DESIGNATION = 'CLERK')


--4. DISPLAY THE LIST OF EMPLOYEES WHO ARE LIVING IN "NEW YORK".
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = (SELECT DEPARTMENT_ID FROM DEPARTMENT 
WHERE LOCATION_ID = (SELECT LOCATION_ID FROM [LOCATION] WHERE CITY = 'NEW YORK'))


--5. FIND OUT NO. OF EMPLOYEES WORKING IN "SALES" DEPARTMENT.
SELECT * FROM EMPLOYEE WHERE DEPARTMENT_ID = (SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME] = 'SALES')


--6. UPDATE THE EMPLOYEES SALARIES, WHO ARE WORKING AS CLERK ON THE BASIS OF 10%.
UPDATE EMPLOYEE
SET SALARY = SALARY*10
WHERE JOB_ID = (SELECT JOB_ID FROM JOB WHERE DESIGNATION = 'CLERK')
SELECT * FROM EMPLOYEE

--7. DELETE THE EMPLOYEES WHO ARE WORKING IN ACCOUNTING DEPARTMENT.
DELETE FROM EMPLOYEE WHERE DEPARTMENT_ID = (SELECT DEPARTMENT_ID FROM DEPARTMENT WHERE [NAME] = 'ACCOUNTING')


--8. DISPLAY THE SECOND HIGHEST SALARY DRAWING EMPLOYEE DETAILS.
SELECT * FROM EMPLOYEE WHERE SALARY = (SELECT MAX(SALARY) FROM EMPLOYEE WHERE SALARY < (SELECT MAX(SALARY) FROM EMPLOYEE))


--9. DISPLAY THE N'TH HIGHEST SALARY DRAWING EMPLOYEE DETAILS.
SELECT DISTINCT E.SALARY FROM EMPLOYEE E WHERE 1=(SELECT COUNT(distinct SALARY) FROM EMPLOYEE WHERE SALARY > E.SALARY)


--10. LIST OUT THE EMPLOYEES WHO EARN MORE THAN EVERY EMPLOYEE IN DEPARTMENT 30.
SELECT COUNT(EMPLOYEE_ID) AS NO_OF_EMPLOYEE FROM EMPLOYEE WHERE SALARY = 
ALL(SELECT SALARY FROM EMPLOYEE WHERE DEPARTMENT_ID = 30) 


--11. LIST OUT THE EMPLOYEES WHO EARN MORE THAN THE LOWEST SALARY IN DEPARTMENT 30.
SELECT COUNT(EMPLOYEE_ID) AS NO_OF_EMPLOYEE FROM EMPLOYEE WHERE SALARY = 
ANY(SELECT SALARY FROM EMPLOYEE WHERE DEPARTMENT_ID = 30) 


--12. FIND OUT WHOSE DEPARTMENT HAS NOT EMPLOYEES.
SELECT DEPARTMENT_ID FROM EMPLOYEE E WHERE
NOT EXISTS(SELECT DEPARTMENT_ID FROM DEPARTMENT D WHERE D.DEPARTMENT_ID = E.DEPARTMENT_ID)


--13. FIND OUT WHICH DEPARTMENT DOES NOT HAVE ANY EMPLOYEES.
SELECT DEPARTMENT_ID FROM EMPLOYEE WHERE DEPARTMENT_ID NOT IN (SELECT DEPARTMENT_ID FROM DEPARTMENT)


--14. FIND OUT THE EMPLOYEES WHO EARN GREATER THAN THE AVERAGE SALARY FOR THEIR DEPARTMENT.
SELECT * FROM EMPLOYEE E WHERE SALARY > (SELECT AVG(SALARY) FROM EMPLOYEE WHERE DEPARTMENT_ID = E.DEPARTMENT_ID)



 