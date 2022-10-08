--Define relations/attributes
/*Attribute relationships are associations  how attributes are connected. 
It defines how tables and columns are joined and used, and which tables are related to other tables. 
Without relationships, there is no interaction between data, and therefore no logical structure.*/

--Define primary keys
/*Primary key in a relational database which is  having unique and not null values for each record. 
It is a unique identifier, such as aadhar number, pan card number.
A relational database must always have one and only one primary key*/


--Create foreign keys

create table role_details
(role_id int primary key,
role_name varchar(100))

insert into role_details values
(1, 'analyst'),
(2, 'accounting')

select * from role_details

create table status_details
(status_id int primary key,
status_name varchar(100),
is_user_working bit)

insert into status_details values
(20, 'working', 0),
(30, 'completed', 1)

select * from status_details

create table user_account 
(id int primary key,
user_name varchar(100),
email varchar(254),
password varchar(200),
password_salt varchar(50),
passwork_hash_algorithm varchar(50))

insert into user_account values
(230,'kevin','abc@gmail.com','abc230@', 'password_salt1', 'algorithm1'),
(231,'priya', 'xyz@gmail.com','xyz231@', 'password_salt2', 'algorithm2')


create table user_has_role
(u_id int primary key,
role_start_time time,
role_end_time time,
user_account_id int foreign key 
references user_account(id),
role_id int foreign key
references role_details(role_id))


insert into user_has_role values
(30,'12:30:00','12:50:00',230,1),
(40,'02:30:00','02:50:00',231,2)


create table user_has_status 
(id int primary key,
status_start_time time,
status_end_time time,
user_account_id int foreign key
references user_account(id),
status_id int foreign key 
references status_details(status_id))

insert into user_has_status values
(50,'05:30:00','07:20:00',230,20),
(70,'07:30:00','09:50:00',231,30)



delete from user_account
delete from role_details
delete from status_details
delete from user_has_status  
delete from user_has_role

select * from role_details
select * from status_details
select * from user_account
select * from user_has_status 
select * from user_has_role

