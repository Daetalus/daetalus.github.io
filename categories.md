---
layout: default
title: "分类：Categories"
---
<ul class="list-unstyled">
{% for cat in site.categories %} 
   <a name="{{ cat[0] }}"></a>
   <h2>{{ cat[0] }}</h2> 
     {% for post in cat[1] %} 
    <li><h4><span>{{ post.date | date:"%Y-%m-%d" }}</span>&nbsp;&nbsp<a href="{{ post.url }}">{{ post.title }}</a></h4></li>
	{% endfor %} 
{% endfor %} 
</ul>

