--*project 1*

--Get all the details from the person table including email ID, phone number, and phone number type
select p1.*,p2.PhoneNumber,p3.EmailAddress,p4.PhoneNumberTypeID from [Person].[Person] p1, [Person].[PersonPhone] p2,
[Person].[EmailAddress] p3, [Person].[PhoneNumberType] p4
where p1.BusinessEntityID = p2.BusinessEntityID and p2.BusinessEntityID = p3.BusinessEntityID and p2.PhoneNumberTypeID = p4.PhoneNumberTypeID


--Get the details of the sales header order made in May 2011
select s.* from [Sales].[SalesOrderHeader] s where year(OrderDate) = 2011 and month(OrderDate) = 05


--Get the details of the sales details order made in the month of May 2011
select s1.*,s2.OrderDate from [Sales].[SalesOrderDetail] s1,[Sales].[SalesOrderHeader] s2 where s1.SalesOrderID = s2.SalesOrderID
and year(OrderDate) = 2011 and month(OrderDate) = 05


--Get the total sales made in May 2011
select sum(OrderQty) as tot_sales from [Sales].[SalesOrderDetail] s1,[Sales].[SalesOrderHeader] s2 where s1.SalesOrderID = s2.SalesOrderID
and year(OrderDate) = 2011 and month(OrderDate) = 05


--Get the total sales made in the year 2011 by month order by increasing sales
select sum(OrderQty) as tot_sales from [Sales].[SalesOrderDetail] s1,[Sales].[SalesOrderHeader] s2 where s1.SalesOrderID = s2.SalesOrderID
and year(OrderDate) = 2011  group by month(OrderDate) order by month(OrderDate) 


--Get the total sales made to the customer with FirstName='Gustavo' and LastName='Achong'
select sum(OrderQty) as tot_sales from [Sales].[SalesOrderDetail] s1,[Sales].[SalesOrderHeader] s2, 
[Sales].[Customer] c,[Sales].[SalesPerson] s3,[Person].[Person] p 
where s1.SalesOrderID = s2.SalesOrderID and s2.CustomerID = c.CustomerID and s3.BusinessEntityID = p.BusinessEntityID
and FirstName='Gustavo' and LastName='Achong' 

